import warnings
warnings.filterwarnings('ignore')
import os
import glob
import h5py
import numpy as np
from scipy.io import loadmat
import cv2
from tqdm import tqdm
import argparse

def gaussian_filter_density(gt, img_shape, sigma=15):
    H, W = img_shape
    density = np.zeros((H, W), dtype=np.float32)
    if len(gt) == 0:
        return density
    gt = np.round(gt).astype(int)
    gt[:, 0] = np.clip(gt[:, 0], 0, W - 1)
    gt[:, 1] = np.clip(gt[:, 1], 0, H - 1)
    for i in range(gt.shape[0]):
        x, y = gt[i]
        density[y, x] += 1
    density = cv2.GaussianBlur(density, (0, 0), sigma)
    return density

def save_density_vis(img, density, save_path):
    dens_vis = density / (density.max() + 1e-6)
    dens_vis = (dens_vis * 255).astype(np.uint8)
    dens_vis = cv2.applyColorMap(dens_vis, cv2.COLORMAP_JET)
    dens_vis = cv2.resize(dens_vis, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 0.6, dens_vis, 0.4, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)

def process_split(root, split, visualize=False):
    img_dir = os.path.join(root, split, "images")
    gt_dir = os.path.join(root, split, "ground_truth")
    mat_files = glob.glob(os.path.join(gt_dir, "GT_*.mat"))
    print(f"[{split}] Found {len(mat_files)} ground truth files")
    for mat_path in tqdm(mat_files):
        mat = loadmat(mat_path)
        pts = mat["locations"][:, :2]
        img_name = os.path.basename(mat_path).replace("GT_", "").replace(".mat", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        density = gaussian_filter_density(pts, (H, W), sigma=15)
        h5_path = mat_path.replace(".mat", ".h5")
        with h5py.File(h5_path, "w") as hf:
            hf["density"] = density
        if visualize:
            vis_root = root.replace("DroneBird", "vis_dronebird")
            vis_path = os.path.join(vis_root, split, "images", img_name)
            save_density_vis(img, density, vis_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='DroneBird')
    parser.add_argument('--input_dir', type=str, default='datasets/DroneBird')
    parser.add_argument('--is_vis', type=bool, default=False)
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    for split in ["train", "val", "test"]:
        process_split(args.input_dir, split, visualize=args.is_vis)