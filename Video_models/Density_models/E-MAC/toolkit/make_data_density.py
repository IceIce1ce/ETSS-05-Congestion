import warnings
warnings.filterwarnings('ignore')
import h5py
import scipy
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy import spatial, ndimage
from multiprocessing import Pool
from functools import partial
import argparse

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2.0 / 2.0 # case: 1 point
        sigma = 6
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode="constant")
    return density

def process(idx, img_paths, total_images, split):
    img_path = img_paths[idx]
    mat_path = img_path.replace(".jpg", ".mat").replace("images", "ground_truth").replace("img", "GT_img")
    mat = io.loadmat(mat_path)
    img = plt.imread(img_path)
    k = np.zeros((int(img.shape[0] / 2), int(img.shape[1] / 2)))
    gt = mat["locations"]
    for i in range(0, len(gt)):
        if int(gt[i][1] / 2) < img.shape[0] / 2 and int(gt[i][0] / 2) < img.shape[1] / 2:
            k[int(gt[i][1] / 2), int(gt[i][0] / 2)] = 1
    k = gaussian_filter_density(k)
    with h5py.File(mat_path.replace("mat", "h5"), "w") as hf:
        hf["density"] = k
    print('[{}]: Idx: [{}/{}], Name: {}'.format(split, idx, total_images, img_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='DroneBird')
    parser.add_argument('--input_dir', type=str, default='datasets/DroneBird')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    list_split = ['train', 'val', 'test']
    for split in list_split:
        img_paths = []
        for img_path in glob.glob(os.path.join(args.input_dir, split, "images", "*.jpg")):
            h5_path = img_path.replace(".jpg", ".h5").replace("images", "ground_truth").replace("img", "GT_img")
            if not os.path.exists(h5_path):
                img_paths.append(img_path)
        img_paths.sort()
        pool = Pool(10)
        partial = partial(process, split=split, img_paths=img_paths, total_images=len(img_paths))
        _ = pool.map(partial, range(len(img_paths)))