import warnings
warnings.filterwarnings("ignore")
import torch
import json
import argparse
from utils.camera_projection import project_points
from utils.append_to_json import append_to_json
import os
import numpy as np
import glob
from scipy.ndimage import binary_dilation

def load_cameras_json(camera_json_path):
    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)
    fx, fy = camera_data["fl_x"], camera_data["fl_y"]
    cx, cy = camera_data["cx"], camera_data["cy"]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera_matrices = []
    image_paths = []
    camera_data["frames"].sort(key=lambda x: x["file_path"])
    for frame in camera_data["frames"]:
        transform_matrix = np.array(frame["transform_matrix"])
        world_to_camera = np.linalg.inv(transform_matrix)
        camera_matrices.append(world_to_camera[:3, :])
        image_paths.append(frame["file_path"])
    camera_matrices = np.array(camera_matrices)
    return camera_matrices, intrinsics, image_paths

def main(args):
    loaded_state = torch.load(args.splatfacto_path, map_location="cpu")
    points = loaded_state["pipeline"]["_model.gauss_params.means"].cpu().numpy()
    opacities = loaded_state["pipeline"]["_model.gauss_params.opacities"].cpu().numpy()
    mask = opacities[:, 0] >= 0
    points = points[mask]
    print("Reading means from 3DGS of shape", points.shape)
    print("Points MAX:", np.max(points, axis=0))
    print("Points MIN:", np.min(points, axis=0))
    raw_depth_files = sorted(glob.glob(os.path.join(args.save_path, "raw_depth", "*.npy")))
    acc_files = sorted(glob.glob(os.path.join(args.save_path, "acc", "*.npy")))
    raw_depths = [np.load(depth) for depth in raw_depth_files]
    accs = [np.load(acc) for acc in acc_files]
    camera_matrices, intrinsics, image_paths = load_cameras_json(args.cameras_path)
    image_files = [os.path.join(args.save_path, im) for im in image_paths]
    assert len(accs) == len(image_files)
    assert len(raw_depths) == len(image_files)
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    scaling = 1 / (max_bounds - min_bounds).max()
    voxel_grid = np.zeros((args.resolution, args.resolution, args.resolution))
    voxel_size = (max_bounds - min_bounds) / args.resolution
    voxel_half_size = np.max(voxel_size) / 2

    def voxel_to_point(i, j, k):
        x = min_bounds[0] + (i + 0.5) * (max_bounds[0] - min_bounds[0]) / args.resolution
        y = min_bounds[1] + (j + 0.5) * (max_bounds[1] - min_bounds[1]) / args.resolution
        z = min_bounds[2] + (k + 0.5) * (max_bounds[2] - min_bounds[2]) / args.resolution
        return np.array([x, y, z])

    voxel_points = []
    voxel_indices = []
    for i in range(args.resolution):
        for j in range(args.resolution):
            for k in range(args.resolution):
                voxel_points.append(voxel_to_point(i, j, k))
                voxel_indices.append((i, j, k))
    voxel_points = np.array(voxel_points)
    voxel_indices = np.array(voxel_indices)
    for cam_idx, (cam_matrix, raw_depth, acc) in enumerate(zip(camera_matrices, raw_depths, accs)):
        projected_2d, depths = project_points(voxel_points, cam_matrix, intrinsics)
        img_h, img_w = raw_depth.shape
        valid_mask = (projected_2d[:, 0] >= 0) & (projected_2d[:, 0] < img_w) & (projected_2d[:, 1] >= 0) & (projected_2d[:, 1] < img_h)
        projected_2d = projected_2d[valid_mask].astype(int)
        depths = depths[valid_mask]
        valid_voxel_indices = voxel_indices[valid_mask]
        count = 0
        for (px, py), voxel_depth, (i, j, k) in zip(projected_2d, depths, valid_voxel_indices):
            pixel_acc = acc[py, px]
            pixel_depth = raw_depth[py, px]
            if pixel_acc < 0.95:
                voxel_grid[i, j, k] += 1
                count += 1
            elif -voxel_depth < pixel_depth:
                voxel_grid[i, j, k] += 1
        print("Added", count, "/", args.resolution**3)
    voxel_grid[voxel_grid >= 5] = -1
    if args.box_thickness > 0:
        voxel_size = (max_bounds - min_bounds) / args.resolution
        thickness_world = (max_bounds[0] - min_bounds[0]) * args.box_thickness
        thickness_voxels = int(np.round(thickness_world / voxel_size[0]))
        print("Thickness:", thickness_voxels)
        if thickness_voxels >= 1:
            voxel_size = (max_bounds - min_bounds) / args.resolution
            thickness_world = (max_bounds[0] - min_bounds[0]) * args.box_thickness
            thickness_voxels = int(np.round(thickness_world / voxel_size[0]))
            inside_voxels = voxel_grid >= 0
            outside_voxels = voxel_grid == -1
            x_transition = np.roll(inside_voxels, shift=-1, axis=0) ^ inside_voxels
            y_transition = np.roll(inside_voxels, shift=-1, axis=1) ^ inside_voxels
            z_transition = np.roll(outside_voxels, shift=-1, axis=2) & inside_voxels
            transition_mask = x_transition | y_transition | z_transition
            struct_element = np.ones((thickness_voxels, thickness_voxels, thickness_voxels))
            dilated_mask = binary_dilation(transition_mask, structure=struct_element)
            voxel_grid[dilated_mask] = -2
    remaining_voxels = np.sum(voxel_grid >= 0)
    total_voxels = np.prod(voxel_grid.shape)
    fraction_remaining = remaining_voxels / total_voxels
    volume = remaining_voxels * np.prod(voxel_size)
    print(f"Fraction of voxels remaining: {fraction_remaining:.4f}")
    print(f"Estimated object volume: {volume:.4f} m3")
    print("Estimated Volume (cm3 or ml):", volume * 1000000)
    append_to_json(args.save_path + "/results.json", "volume_in_cm3", volume * 1000000)
    append_to_json(args.save_path + "/results.json", "volume_in_m3", volume)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splatfacto_path", type=str, default='')
    parser.add_argument("--cameras_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--box-thickness", type=float, required=True)
    parser.add_argument('--resolution', type=int, default=100)
    args = parser.parse_args()
    main(args)