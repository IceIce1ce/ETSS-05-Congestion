from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse

def cal_new_size1(im_h, im_w, resize_h, resize_w):
    ratio_h = 1.0 * resize_h/im_h
    ratio_w = 1.0 * resize_w/im_w
    return ratio_h, ratio_w

def find_dis(point):
    square = np.sum(point * points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2 * np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path, resize_h, resize_w):
    im = Image.open(im_path)
    im_w, im_h = im.size
    points = loadmat(im_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))['image_info'][0][0][0][0][0].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    rr_h, rr_w = cal_new_size1(im_h, im_w, resize_h, resize_w)
    im = cv2.resize(np.array(im), (resize_w, resize_h), cv2.INTER_CUBIC)
    points[:, 0] = points[:, 0] * rr_w
    points[:, 1] = points[:, 1] * rr_h
    im = np.array(im)
    return Image.fromarray(im), points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', default='data/ShanghaiTech/part_A', type=str)
    parser.add_argument('--output_dir', default='data/sha', type=str)
    parser.add_argument('--resize_w', type=int, default=256)
    parser.add_argument('--resize_h', type=int, default=256)
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    for phase in ['train_data', 'test_data']:
        sub_dir = os.path.join(args.input_dir, phase)
        sub_dir_im = os.path.join(sub_dir, 'images')
        save_dir = os.path.join(args.output_dir, phase)
        sub_save_dir = os.path.join(save_dir, 'images')
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        for im_path in glob(os.path.join(sub_dir_im, '*.jpg')):
            name = os.path.basename(im_path)
            im, points = generate_data(im_path, args.resize_h, args.resize_w) # [256, 256, 3], [77, 2]
            dis = find_dis(points) # [77, 1]
            points = np.concatenate((points, dis), axis=1) # [77, 3]
            im_save_path = os.path.join(sub_save_dir, name)
            im.save(im_save_path)
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)