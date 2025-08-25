import scipy.io as io
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import h5py
import numpy as np
import os

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32) # [685, 1024]
    gt_count = np.count_nonzero(gt) # [321]
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))) # [321, 2]
    leafsize = 2048
    tree = scipy.spatial.cKDTree(pts.copy(), leafsize=leafsize) # default: KDTree
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32) # [685, 1024]
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1 # [1]
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant') # [685, 1024]
    return density

if __name__ == '__main__':
    root = 'data/ShanghaiTech'
    part_A_train = os.path.join(root, 'part_A/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B/test_data', 'images')
    path_sets = [part_A_train, part_A_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path) # [685, 1024, 3]
        k = np.zeros((img.shape[0], img.shape[1])) # [685, 1024]
        gt = mat["image_info"][0, 0][0, 0][0] # [321, 2]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density(k) # [685, 1024]
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:
            hf['density'] = k
    path_sets = [part_B_train, part_B_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path) # [768, 1024, 3]
        k = np.zeros((img.shape[0], img.shape[1])) # [768, 1024]
        gt = mat["image_info"][0, 0][0, 0][0] # [143, 2]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter(k, 15) # [768, 1024]
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:
            hf['density'] = k