import numpy as np
import scipy.ndimage
import scipy.io as scio
from PIL import Image
import os
import glob
import cv2
import argparse
import warnings
warnings.filterwarnings("ignore")

def generate_data(is_max, is_rand, H_ratio=0.5, W_ratio=0.2, mask_ID='2020_12_16_v2_rand_', args=None):
    raw_data_root = os.path.join(args.input_dir, 'train_data/')
    raw_matdata_root = os.path.join(raw_data_root, 'ground-truth/')
    raw_image_root = os.path.join(raw_data_root, 'images/')
    new_data_root = raw_data_root.replace('ShanghaiTech','masked_data')
    new_maskmap_root = os.path.join(new_data_root, mask_ID, 'mask/')
    gt_paths = []
    for gt_path in glob.glob(os.path.join(raw_matdata_root, '*.mat')):
        gt_paths.append(gt_path)
        idx = gt_path.split('/')[-1].split('_')[-1].split('.')[0]
        gt_file = raw_matdata_root + 'GT_IMG_' + str(idx) + '.mat'
        image_file = raw_image_root + 'IMG_' + str(idx) + '.jpg'
        new_name = 'GT_IMG_' + str(idx) + '.h5'
        gt = scio.loadmat(gt_file)
        if args.type_dataset == 'sha' or args.type_dataset == 'shb':
            gt = gt['image_info'][0][0][0][0][0]
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        image = Image.open(image_file)
        image = np.array(image) # [685, 1024, 3]
        H, W = image.shape[0], image.shape[1]
        gt_map = np.zeros((H, W)) # [685, 1024]
        for j, (x, y) in enumerate(gt):
            if x > W or y > H:
                continue
            gt_map[int(y), int(x)] = 1
        density_map_sigma15 = scipy.ndimage.filters.gaussian_filter(gt_map, 15) # [685, 1024]
        mask = np.zeros((H, W)) # [685, 1024]
        d_x = int(W * W_ratio)
        d_y = int(H * H_ratio)
        if W_ratio < 1:
            if is_max:
                (pos_y, pos_x) = np.unravel_index(np.argmax(density_map_sigma15), density_map_sigma15.shape)
            if is_rand:
                if H - d_x - 10 < 10:
                    pos_x = np.random.randint(0, W - d_x - 10)
                    pos_y = np.random.randint(0, H - d_y - 10)
                else:
                    pos_x = np.random.randint(10, W - d_x - 10)
                    pos_y = np.random.randint(10, H - d_y - 10)
            if pos_y >= image.shape[0] - d_y:
                pos_y = image.shape[0] - d_y
            if pos_y <= 0:
                pos_y = 0
            if pos_x >= image.shape[1] - d_x:
                pos_x = image.shape[1] - d_x
            if pos_x <= 0:
                pos_x = 0
            mask[pos_y:pos_y+d_y, pos_x:pos_x+d_x] = 1
        else:
            mask = np.ones((H,W))
        mask = mask > 0 # [685, 1024]
        mask = np.array(mask + 0) # [685, 1024]
        mask_img_name = os.path.join(new_maskmap_root, new_name.replace('.h5','.png'))
        if not os.path.exists(new_maskmap_root):
            os.makedirs(new_maskmap_root)
        mask = (mask * 255).astype(np.uint8) # [685, 1024]
        cv2.imwrite(mask_img_name, mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A')
    parser.add_argument('--mask_ratio', type=int, default=10)
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    if args.mask_ratio == 1:
        H_ratio = 0.1
        W_ratio = 0.1
        mask_ID = '2020_12_19_r01_rand_'
        is_max, is_rand = False, True
        generate_data(is_max, is_rand, H_ratio, W_ratio, mask_ID, args=args)
    elif args.mask_ratio == 10:
        is_max, is_rand = False, True
        generate_data(is_max, is_rand, args=args)
    elif args.mask_ratio == 25:
        H_ratio = 0.5
        W_ratio = 0.5
        mask_ID = '2020_12_19_r25_rand_'
        is_max, is_rand = False, True
        generate_data(is_max, is_rand, H_ratio, W_ratio, mask_ID, args=args)
    elif args.mask_ratio == 50:
        is_max, is_rand = False, True
        H_ratio, W_ratio = 0.7, 0.7
        mask_ID = '2020_12_19_r50_rand_'
        generate_data(is_max, is_rand, H_ratio, W_ratio, mask_ID, args=args)
    else:
        print('This mask ratio does not exist')
        raise NotImplementedError