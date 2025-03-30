import glfw
import numpy as np
import OpenGL.GL as gl
import cv2
import pickle

import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, GaussianStreamManager
from utils.pose_utils import generate_ellipse_path, getWorld2View2
from generate_cluster import collect_features

import time

SPARSE_ADAM_AVAILABLE = False

class OpenGLVideoPlayer:
    def __init__(self, H, W, total_frames):
        """
        Initializes the video player by loading images and setting up OpenGL.
        """
        self.HEIGHT, self.WIDTH = H, W
        self.FRAMES = total_frames

        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed!")

        self.window = glfw.create_window(self.WIDTH, self.HEIGHT, "OpenGL Video Player", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window!")

        glfw.make_context_current(self.window)

        # OpenGL texture setup
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # Initialize an empty black frame
        blank_frame = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.WIDTH, self.HEIGHT, 0, gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE, blank_frame)
    
    def draw_texture(self):
        """Renders the current OpenGL texture as a fullscreen quad."""
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0)
        gl.glVertex2f(-1, -1)
        gl.glTexCoord2f(1, 0)
        gl.glVertex2f(1, -1)
        gl.glTexCoord2f(1, 1)
        gl.glVertex2f(1, 1)
        gl.glTexCoord2f(0, 1)
        gl.glVertex2f(-1, 1)
        gl.glEnd()

        gl.glDisable(gl.GL_TEXTURE_2D)
    
    def draw_progress_bar(self, progress):
        """
        Draws a green progress bar at the bottom of the window.
        :param progress: Float between 0 and 1 representing the progress percentage.
        """
        gl.glPushAttrib(gl.GL_CURRENT_BIT)  # Save current color state

        gl.glColor3f(0.0, 1.0, 0.0)  # Green color for progress bar
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(-1, -0.9)
        gl.glVertex2f(-1 + 2 * progress, -0.9)
        gl.glVertex2f(-1 + 2 * progress, -0.85)
        gl.glVertex2f(-1, -0.85)
        gl.glEnd()

        gl.glPopAttrib()  # Restore previous color state

    def add_text_to_frame(self, frame, text, position):
        """
        Uses OpenCV to draw text on the video frame.
        :param frame: The video frame (numpy array)
        :param text: The text to draw
        :param position: (x, y) coordinates for the text position
        :return: Frame with text drawn on it
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        color = (255, 255, 255)  # White text
        
        cv2.putText(frame, text, position, font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)
        return frame

    def run(self, frame_idx, fps, frame_data):
        glfw.poll_events()

        if isinstance(frame_data, torch.Tensor):
            frame_data = frame_data.to("cpu").detach().numpy()

        if frame_data.shape[0] == 3:  
            frame_data = frame_data.transpose(1, 2, 0)

        frame_data = (frame_data * 255).astype(np.uint8) if frame_data.dtype != np.uint8 else frame_data

        progress_text = f"Frame: {frame_idx}/{self.FRAMES} | FPS: {fps:.2f}"
        frame_data = self.add_text_to_frame(np.ascontiguousarray(frame_data), progress_text, (50, 80))

        frame_data = np.ascontiguousarray(cv2.flip(frame_data, 0))
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.WIDTH, self.HEIGHT,
                            gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame_data)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.draw_texture()
        self.draw_progress_bar(frame_idx / self.FRAMES)

        glfw.swap_buffers(self.window)


def poses2views_like(template, poses):
    import copy
    views = []
    for pose in poses:
        view = copy.deepcopy(template)
        view.R = pose[:3, :3].T
        view.T = pose[:3, 3]
        view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        views.append(view)
    return views

def predict(X, centers):
    distances = np.sum((X[:, np.newaxis, :] - centers) ** 2,axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

def render_set(model_path, views, gaussians, pipeline, background, train_test_exp, separate_sh, args):    
    total_frame = args.frames
    poses = generate_ellipse_path(views, total_frame)
    test_views = poses2views_like(views[0], poses)
    
    stream_manager = None
    if args.load_seele:
        # Load cluster data
        with open(os.path.join(model_path, "clusters", "clusters.pkl"), "rb") as f:
            cluster_data = pickle.load(f)
        K = len(cluster_data["cluster_viewpoint"])
        cluster_centers = cluster_data["centers"]
        
        # Determine the test cluster labels
        test_features = collect_features(test_views)
        test_labels = predict(test_features, cluster_centers)
        
        # Load all Gaussians to CPU
        cluster_gaussians = [
            torch.load(os.path.join(model_path, f"clusters/finetune/point_cloud_{cid}.pth"), map_location="cpu")
            for cid in range(K)
        ]

        # Initialize stream manager
        stream_manager = GaussianStreamManager(
            cluster_gaussians=cluster_gaussians,
            initial_cid=test_labels[0]
        )
        
    # Warm up
    for _ in range(20):
        render(views[0], gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
    
    player = OpenGLVideoPlayer(W=views[0].image_width, H=views[0].image_height, total_frames=total_frame)
    start_time = time.time()
    for idx, view in enumerate(tqdm(test_views, desc="frame_data progress")):
        if glfw.window_should_close(player.window): 
            break
        
        if args.load_seele:
            if idx + 1 < len(test_views):
                next_cid = test_labels[idx+1]
                stream_manager.preload_next(next_cid)

            gaussians.restore_gaussians(stream_manager.get_current())

            rendering = render(
                view, gaussians, pipeline, background,
                use_trained_exp=train_test_exp,
                separate_sh=separate_sh,
                rasterizer_type="CR"
            )["render"]
                
            torch.cuda.current_stream().wait_stream(stream_manager.load_stream)
            stream_manager.switch_gaussians()
        else:
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
                    
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        start_time = end_time
        
        # print(fps)
        player.run(idx, fps, rendering)
        
    glfw.terminate()
    if stream_manager is not None:
        stream_manager.cleanup()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, separate_sh: bool, args: ArgumentParser):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)

# Example usage
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--frames", default=1000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_seele", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), SPARSE_ADAM_AVAILABLE, args)
