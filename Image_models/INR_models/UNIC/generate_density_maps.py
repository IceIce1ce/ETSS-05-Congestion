# https://github.com/davideverona/deep-crowd-counting_crowdnet
import h5py
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import argparse

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape)) // 2. // 2.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', default='data/sha', type=str)
    args = parser.parse_args()

    print('Generate density maps for dataset:', args.type_dataset)
    RSOC_train = os.path.join(args.input_dir, 'train_data/images')
    os.makedirs(os.path.join(args.input_dir, 'train_data/ground_truth'), exist_ok=True)
    RSOC_test = os.path.join(args.input_dir, 'test_data/images')
    os.makedirs(os.path.join(args.input_dir, 'test_data/ground_truth'), exist_ok=True)
    img_paths = []
    for path in [RSOC_train, RSOC_test]:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    train_list = []
    test_list = []
    for img_path in glob.glob(os.path.join(RSOC_train, '*.jpg')):
        train_list.append(img_path)
    for img_path in glob.glob(os.path.join(RSOC_test, '*.jpg')):
        test_list.append(img_path)
    for img_path in img_paths:
        gd_path = img_path.replace('jpg', 'npy')
        gt = np.load(gd_path, allow_pickle=True).astype(np.float32) # [77, 3]
        gt = gt[:, :2]
        img = plt.imread(img_path) # [256, 256, 3]
        k = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0,len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter(k, 4) # [256, 256]
        # k = gaussian_filter_density(k) # [256, 256]
        groundtruth = np.asarray(k) # [256, 256]
        with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k