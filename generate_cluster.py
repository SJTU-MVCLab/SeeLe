#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2

import joblib
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as Rot

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def generate_features_from_Rt(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    # R_w2c: R.T, t_w2c: t 
    # R_c2w: R, t_c2w: -R.T @ t
    w2c = getWorld2View2(R, t, translate=translate, scale=scale)
    c2w = np.linalg.inv(w2c)
    
    rot = Rot.from_matrix(c2w[:3, :3]) # This function will orthonormalize R automatically.
    q = rot.as_quat(canonical=True) 
    feature_vector = np.concatenate([c2w[:3, 3], q])
    return feature_vector

def extract_features(views):
    features = []
    for view in views:
        features.append(generate_features_from_Rt(view.R, view.T))
    features = np.stack(features, axis=0)
    return features

def merge_neighbor_mask(centers, cluster_masks, labels, neigh):
    K, P = cluster_masks.shape
    
    total_shared = total_exclusive = 0
    merge_gaussians, merge_viewpoint = [], []
    cluster_masks = cluster_masks.astype(np.uint32)
    average_gaussians = 0
    for cid in range(K):
        base = centers[cid:cid+1]
        dist2 = np.square(base - centers).sum(1)
        merge_clusters = np.argsort(dist2)[:neigh + 1]

        viewpoints = np.concatenate([(labels == cluster).nonzero()[0] for cluster in merge_clusters])
        merge_viewpoint.append(viewpoints)

        gaussians_counter = cluster_masks[merge_clusters].sum(axis=0)
        shared_mask = (gaussians_counter > ((neigh + 1) // 2))
        exclusive_mask = np.logical_xor(shared_mask, (gaussians_counter != 0))

        shared, exclusive = map(lambda x: x.nonzero()[0], [shared_mask, exclusive_mask])
        gaussian_ids = np.concatenate([shared, exclusive], axis=0)

        lens = (len(shared), len(exclusive))
        merge_gaussians.append((gaussian_ids, lens))
        
        total_shared += lens[0]
        total_exclusive += lens[1]
        average_gaussians += lens[0] + lens[1]
        
    total_shared //= K
    total_exclusive //= K
    average_gaussians //= K
    print(f"Total gaussians: {P}, average shared gaussians: {total_shared}, average exclusive gaussians: {total_exclusive}, average number of gaussians: {average_gaussians}")
    print(f"Expansion ratio: {(total_exclusive + total_shared) / P}")
    return merge_gaussians, merge_viewpoint

def render_set(views, gaussians, pipeline, background, train_test_exp, separate_sh):
    gaussian_masks = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, rasterizer_type="Mark")
        visible_gaussians = out["visible_gaussians"].cpu().numpy()
        gaussian_masks.append(visible_gaussians != 0)
        
    return np.stack(gaussian_masks, axis=0)
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, args, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        train_features = extract_features(scene.getTrainCameras())
        test_features = extract_features(scene.getTestCameras())
        kmeans = KMeans(n_clusters=args.k, random_state=42, n_init='auto').fit(train_features)
        centers = kmeans.cluster_centers_
        train_labels = kmeans.labels_
        test_labels = kmeans.predict(test_features)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view_gaussian_masks = render_set(scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
        cluster_gaussian_masks = np.stack([np.any(view_gaussian_masks[train_labels == j], axis=0) for j in range(args.k)], axis=0)
        merge_gaussians, merge_viewpoint = merge_neighbor_mask(centers, cluster_gaussian_masks, train_labels, neigh=args.n)
        
        save_path = os.path.join(dataset.model_path, "clusters")
        makedirs(save_path, exist_ok=True)
        data = {
            "cluster_gaussians": merge_gaussians, 
            "cluster_viewpoint": merge_viewpoint,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "centers": centers,
        }
        joblib.dump(data, os.path.join(save_path, "clusters.pkl"))
        joblib.dump(kmeans,  os.path.join(save_path, "kmeans_model.pkl"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-k", type=int, default = 24)
    parser.add_argument("-n", type=int, default = 4)
    args = get_combined_args(parser)
    print("Generating clusters for" + args.model_path)
    print(f"k: {args.k}, n: {args.n}")
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args, SPARSE_ADAM_AVAILABLE)
