import pyglet
import numpy as np
import joblib
import torch
import os
import time
from tqdm import tqdm
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import render, GaussianModel, GaussianStreamManager
from utils.general_utils import safe_state
from utils.pose_utils import generate_ellipse_path, getWorld2View2
from arguments import ModelParams, PipelineParams, get_combined_args
from generate_cluster import generate_features_from_Rt
import torchvision
SPARSE_ADAM_AVAILABLE = False

class VideoPlayer:
    """Efficient video player using pyglet for 3DGS rendering display."""
    
    def __init__(self, width: int, height: int, total_frames: int):
        """Initialize the video player window and UI elements.
        
        Args:
            width: Width of the video frame
            height: Height of the video frame
            total_frames: Total number of frames to be displayed
        """
        self.window = pyglet.window.Window(
            width=width, 
            height=height, 
            caption='3DGS Rendering Viewer'
        )
        self.total_frames = total_frames
        self.current_frame = 0
        self.fps = 0.0
        self.last_time = time.time()
        
        # Initialize texture with blank frame
        self._init_texture(width, height)
        
        # Setup UI elements
        self._setup_ui(width, height)
        
        # Register event handlers
        self.window.event(self.on_draw)

    def _init_texture(self, width: int, height: int):
        """Initialize the OpenGL texture with blank data."""
        blank_data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
        self.texture = pyglet.image.ImageData(
            width, height, 'RGB', blank_data
        ).get_texture()

    def _setup_ui(self, width: int, height: int):
        """Initialize UI components (FPS counter and progress bar)."""
        self.batch = pyglet.graphics.Batch()
        
        # Frame counter label
        self.label = pyglet.text.Label(
            '', 
            x=10, y=height-30,
            font_size=16,
            color=(255, 255, 255, 255),
            batch=self.batch
        )
        
        # Progress bar (positioned at bottom with 2% margin)
        self.progress_bar = pyglet.shapes.Rectangle(
            x=width*0.01, y=5, 
            width=0, height=10,
            color=(0, 255, 0),
            batch=self.batch
        )
        self.progress_bar_max_width = width*0.98

    def update_frame(self, frame_data: np.ndarray):
        """Update the display with new frame data.
        
        Args:
            frame_data: Numpy array containing frame data (H,W,3)
        """
        # Convert tensor if necessary
        if isinstance(frame_data, torch.Tensor):
            frame_data = frame_data.detach().cpu().numpy()
        
        # Ensure correct shape and type
        if frame_data.shape[0] == 3:  # CHW to HWC
            frame_data = frame_data.transpose(1, 2, 0)
        if frame_data.dtype != np.uint8:
            frame_data = (frame_data * 255).astype(np.uint8)
        
        # Flip vertically and update texture
        frame_data = np.ascontiguousarray(np.flipud(frame_data))
        self.texture = pyglet.image.ImageData(
            self.window.width, self.window.height,
            'RGB', frame_data.tobytes()
        ).get_texture()

        # Update performance metrics
        self._update_perf_metrics()
        
        # Update UI
        self.label.text = f'Frame: {self.current_frame+1}/{self.total_frames} | FPS: {self.fps:.2f}'
        self.progress_bar.width = self.progress_bar_max_width * (self.current_frame+1)/self.total_frames
        self.current_frame += 1

    def _update_perf_metrics(self):
        """Calculate and update FPS metrics."""
        current_time = time.time()
        self.fps = 1.0 / max(0.001, current_time - self.last_time)  # Avoid division by zero
        self.last_time = current_time

    def on_draw(self):
        """Window draw event handler."""
        self.window.clear()
        if self.texture:
            self.texture.blit(0, 0, width=self.window.width, height=self.window.height)
        self.batch.draw()

def predict(X, centers):
    distances = np.sum((X[:, np.newaxis, :] - centers) ** 2,axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

def extract_features(Rt_list, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
    features = []
    for (R, t) in Rt_list:
        features.append(generate_features_from_Rt(R, t, trans, scale))
    return np.stack(features, axis=0)

def render_set(model_path, views, gaussians, pipeline, background, train_test_exp, separate_sh, args):    
    total_frame = args.frames
    load_seele = args.load_seele
    use_gui = args.use_gui
    
    # prepare the views
    poses = generate_ellipse_path(views, total_frame)
    Rt_list = [(pose[:3, :3].T, pose[:3, 3]) for pose in poses]
    w2c_list = [
        torch.tensor(getWorld2View2(Rt_list[frame][0], Rt_list[frame][1], views[0].trans, views[0].scale)).transpose(0, 1).cuda() 
        for frame in range(total_frame)
    ]
    
    stream_manager, labels = None, None
    if load_seele:
        # Load cluster data
        cluster_data = joblib.load(os.path.join(model_path, "clusters", "clusters.pkl"))
        K = len(cluster_data["cluster_viewpoint"])
        cluster_centers = cluster_data["centers"]
        
        # Determine the test cluster labels
        test_features = extract_features(Rt_list, trans=views[0].trans, scale=views[0].scale)
        labels = predict(test_features, cluster_centers)
        
        # Load all Gaussians to CPU
        cluster_gaussians = [
            torch.load(os.path.join(model_path, f"clusters/finetune/point_cloud_{cid}.pth"), map_location="cpu")
            for cid in range(K)
        ]
        
        # Initialize stream manager
        stream_manager = GaussianStreamManager(
            cluster_gaussians=cluster_gaussians,
            initial_cid=labels[0]
        )
        
    # Warm up
    for _ in range(5):
        render(views[0], gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        
    def render_view(frame):
        view = views[0]
        view.world_view_transform = w2c_list[frame]
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        
        if load_seele:
            # Preload next frame's Gaussians
            if frame + 1 < total_frame:
                next_cid = labels[frame + 1]
                stream_manager.preload_next(next_cid)
                
            # Restore current Gaussians and render
            gaussians.restore_gaussians(stream_manager.get_current())
            rendering = render(
                view, gaussians, pipeline, background,
                use_trained_exp=train_test_exp,
                separate_sh=separate_sh,
                rasterizer_type="CR"
            )["render"]
            
            # Synchronize streams and switch buffers
            stream_manager.switch_gaussians()
        else:
            # Standard rendering path
            rendering = render(
                view, gaussians, pipeline, background,
                use_trained_exp=train_test_exp,
                separate_sh=separate_sh
            )["render"]
            
        return rendering
    
    if use_gui:
        # Initialize video player
        player = VideoPlayer(width=views[0].image_width, height=views[0].image_height, total_frames=total_frame)

        def update_frame(dt):
            """Callback function for frame updates."""
            nonlocal stream_manager, gaussians
    
            if player.current_frame >= args.frames - 1:
                pyglet.app.exit()
                return
            
            rendering = render_view(player.current_frame)
            # Update display
            player.update_frame(rendering)
        
        # Start rendering loop (target 500 FPS)
        pyglet.clock.schedule_interval(update_frame, 1/500.0)
        pyglet.app.run()
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        for frame_idx in range(total_frame):
            if load_seele:
                print(f"Rendering {frame_idx} image belong to cluster {labels[frame_idx]}")
            else:
                print(f"Rnedering {frame_idx} image")           
            rendering = render_view(frame_idx)
            torchvision.utils.save_image(rendering, os.path.join(output_dir, '{0:05d}'.format(frame_idx) + ".png"))
        
    # clean up
    if stream_manager is not None:
        stream_manager.cleanup()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, separate_sh: bool, args: ArgumentParser):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)

# Example usage
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--frames", default=200, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_seele", action="store_true")
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument('--output_dir', type=str, default="output/videos")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), SPARSE_ADAM_AVAILABLE, args)
