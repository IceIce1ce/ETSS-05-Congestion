import scipy.io as io
import os
import glob
from matplotlib import pyplot as plt
import argparse
import numpy as np

def save_gt_points_to_txt(gt_points, txt_path):
    with open(txt_path, 'w') as f:
        for point in gt_points:
            f.write(f"{point[0]} {point[1]}\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    part_A_train = os.path.join(args.input_dir, 'train_data', 'images')
    part_A_test = os.path.join(args.input_dir, 'test_data', 'images')
    path_sets = [part_A_train, part_A_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path) # [685, 1024, 3]
        k = np.zeros((img.shape[0], img.shape[1])) # [685, 1024]
        gt = mat["image_info"][0, 0][0, 0][0] # [321, 2]
        save_gt_points_to_txt(gt, img_path.replace('.jpg', '.txt'))