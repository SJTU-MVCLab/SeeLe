import pyglet
import numpy as np
import joblib
import torch
import os
import copy
import time
from tqdm import tqdm
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import render, GaussianModel, GaussianStreamManager
from utils.general_utils import safe_state
from utils.pose_utils import generate_ellipse_path, getWorld2View2
from arguments import ModelParams, PipelineParams, get_combined_args
from generate_cluster import collect_features

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
            frame_idx: Current frame index (0-based)
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
        
        # Update UI
        self.label.text = f'Frame: {self.current_frame + 1}/{self.total_frames} | FPS: {self.fps:.2f}'
        self.progress_bar.width = self.progress_bar_max_width * (self.current_frame+1)/self.total_frames
        print(self.label.text)

    def on_draw(self):
        """Window draw event handler."""
        self.window.clear()
        if self.texture:
            self.texture.blit(0, 0, width=self.window.width, height=self.window.height)
        self.batch.draw()


def create_views_from_poses(template_view, poses):
    """Generate camera views from pose matrices.
    
    Args:
        template_view: Template camera view to copy parameters from
        poses: List of pose matrices (4x4)
        
    Returns:
        List of camera view objects
    """
    views = []
    for pose in poses:
        view = copy.deepcopy(template_view)
        view.R = pose[:3, :3].T
        view.T = pose[:3, 3]
        view.world_view_transform = torch.tensor(
            getWorld2View2(view.R, view.T, view.trans, view.scale)
        ).transpose(0, 1).cuda()
        view.full_proj_transform = (
            view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        views.append(view)
    return views


def predict_cluster_labels(features, centers):
    """Predict cluster labels for given features using nearest center.
    
    Args:
        features: Input features (N,D)
        centers: Cluster centers (K,D)
        
    Returns:
        Array of cluster labels (N,)
    """
    distances = np.sum((features[:, np.newaxis, :] - centers) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def render_animation(
    model_path: str,
    input_views: list,
    gaussians: GaussianModel,
    pipeline_params: dict,
    background: torch.Tensor,
    use_trained_exp: bool,
    separate_sh: bool,
    args
):
    """Main rendering function for 3DGS animation.
    
    Args:
        model_path: Path to the trained model
        input_views: List of input camera views
        gaussians: GaussianModel instance
        pipeline_params: Rendering pipeline parameters
        background: Background color tensor
        use_trained_exp: Whether to use trained exposures
        separate_sh: Whether to use separate spherical harmonics
        args: Command line arguments
    """
    # Generate camera path and corresponding views
    poses = generate_ellipse_path(input_views, args.frames)
    render_views = create_views_from_poses(input_views[0], poses)
    
    # Initialize streaming manager if using SEELE
    stream_manager = None
    if args.load_seele:
        cluster_data = joblib.load(os.path.join(model_path, "clusters", "clusters.pkl"))
        
        # Predict cluster labels for each view
        view_features = collect_features(render_views)
        view_labels = predict_cluster_labels(view_features, cluster_data["centers"])
        
        # Load all cluster Gaussians
        cluster_gaussians = [
            torch.load(
                os.path.join(model_path, f"clusters/finetune/point_cloud_{cid}.pth"),
                map_location="cpu"
            )
            for cid in range(len(cluster_data["cluster_viewpoint"]))
        ]
        
        stream_manager = GaussianStreamManager(
            cluster_gaussians=cluster_gaussians,
            initial_cid=view_labels[0]
        )
    
    # Warm-up render passes
    for _ in range(3):  # Reduced from 20 to 3 for faster startup
        render(
            input_views[0], gaussians, pipeline_params, background,
            use_trained_exp=use_trained_exp,
            separate_sh=separate_sh
        )
    
    # Initialize video player
    player = VideoPlayer(
        width=input_views[0].image_width,
        height=input_views[0].image_height,
        total_frames=args.frames
    )

    def update_frame(dt):
        """Callback function for frame updates."""
        nonlocal stream_manager, gaussians
        
        if player.current_frame >= args.frames:
            pyglet.app.exit()
            return
        
        t_start = time.time()
        current_view = render_views[player.current_frame]
        
        # Handle streaming if enabled
        if args.load_seele:
            # Preload next frame's Gaussians
            if player.current_frame + 1 < args.frames:
                next_cid = view_labels[player.current_frame + 1]
                stream_manager.preload_next(next_cid)
            
            # Restore current Gaussians and render
            gaussians.restore_gaussians(stream_manager.get_current())
            rendering = render(
                current_view, gaussians, pipeline_params, background,
                use_trained_exp=use_trained_exp,
                separate_sh=separate_sh,
                rasterizer_type="CR"
            )["render"]
            
            # Synchronize streams and switch buffers
            torch.cuda.current_stream().wait_stream(stream_manager.load_stream)
            stream_manager.switch_gaussians()
        else:
            # Standard rendering path
            rendering = render(
                current_view, gaussians, pipeline_params, background,
                use_trained_exp=use_trained_exp,
                separate_sh=separate_sh
            )["render"]
            
        torch.cuda.synchronize()
        t_end = time.time()
        player.fps = 1.0 / (t_end - t_start)
        
        # Update display
        player.update_frame(rendering)
        player.current_frame += 1
    
    # Start rendering loop
    pyglet.clock.schedule_interval(update_frame, 1/1000.0)
    pyglet.app.run()
    
    # Cleanup
    if stream_manager is not None:
        stream_manager.cleanup()


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    separate_sh: bool,
    args: ArgumentParser
):
    """Main entry point for rendering sets."""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_animation(
            dataset.model_path,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
            dataset.train_test_exp,
            separate_sh,
            args
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser(description="3DGS Rendering Parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--frames", default=1000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_seele", action="store_true")
    args = get_combined_args(parser)
    
    print(f"Rendering {args.model_path}")
    safe_state(args.quiet)
    
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        SPARSE_ADAM_AVAILABLE,
        args
    )