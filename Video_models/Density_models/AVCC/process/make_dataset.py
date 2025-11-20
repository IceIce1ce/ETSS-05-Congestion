import glob
import json
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import os
import h5py
import argparse

def main(args):
    WIDTH, HEIGHT = 640, 360
    train_folder = os.path.join(args.input_dir, 'train_data')
    test_folder = os.path.join(args.input_dir, 'test_data')
    path_sets = ([os.path.join(train_folder, f) for f in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, f))] +
                 [os.path.join(test_folder, f) for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, f))])
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        gt_path = img_path.replace('.jpg', '.json')
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        anno_list = list(gt.values())[0]['regions']
        img = plt.imread(img_path) # [1080, 1920, 3]
        k = np.zeros((HEIGHT, WIDTH)) # [360, 640]
        rate_h = img.shape[0] / HEIGHT
        rate_w = img.shape[1] / WIDTH
        for i in range(0, len(anno_list)):
            y_anno = min(int(anno_list[i]['shape_attributes']['y'] / rate_h), HEIGHT)
            x_anno = min(int(anno_list[i]['shape_attributes']['x'] / rate_w), WIDTH)
            k[y_anno, x_anno] = 1
        k = gaussian_filter(k, 3) # [360, 640]
        with h5py.File(img_path.replace('.jpg', '_resize.h5'), 'w') as hf:
            hf['density'] = k
            hf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='FDST')
    parser.add_argument('--input_dir', type=str, default='datasets/FDST')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)