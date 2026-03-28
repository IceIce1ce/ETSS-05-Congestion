import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from scipy.io import loadmat
import argparse

def main(args):
    splits = ['train_data', 'test_data']
    for split in splits:
        img_dir = os.path.join(args.input_dir, split, 'images')
        gt_dir = os.path.join(args.input_dir, split, 'ground-truth')
        new_anno_dir = os.path.join(args.input_dir, split, 'new-anno')
        if not os.path.exists(new_anno_dir):
            os.makedirs(new_anno_dir)
        img_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        for img_name in img_list:
            img_id = img_name.split('.')[0]
            mat_path = os.path.join(gt_dir, f'GT_{img_id}.mat')
            new_anno_path = os.path.join(new_anno_dir, f'GT_{img_id}.npy')
            mat = loadmat(mat_path)
            points = mat['image_info'][0][0][0][0][0]
            np.save(new_anno_path, points.astype(np.float32))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)
