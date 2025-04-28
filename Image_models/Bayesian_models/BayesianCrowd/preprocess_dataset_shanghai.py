from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    mat_path = 'datasets/ShanghaiTech/part_A_final/train_data/ground_truth/'
    im = Image.open(im_path)
    im_w, im_h = im.size
    im_path = im_path.split('/')[-1]
    mat_path = mat_path + "GT_" + im_path.replace('.jpg', '.mat')
    points = loadmat(mat_path)['image_info'][0, 0][0, 0][0]
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--origin-dir', default='datasets/ShanghaiTech/part_A_final/train_data/images/')
    parser.add_argument('--data-dir', default='datasets/processed_shanghaitech_part_a')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048
    im_list = glob(os.path.join(args.origin_dir, '*jpg'))
    train_data = im_list[:250]
    val_data = im_list[250:]
    train_save_dir = os.path.join(save_dir, 'train')
    val_save_dir = os.path.join(save_dir, 'val')
    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)
    for i, path in enumerate(train_data):
        name = os.path.basename(path)
        print(path)
        im, points = generate_data(path)
        dis = find_dis(points)
        points = np.concatenate((points, dis), axis=1)
        im_save_path = os.path.join(train_save_dir, name)
        im.save(im_save_path)
        gd_save_path = im_save_path.replace('jpg', 'npy')
        np.save(gd_save_path, points)
    for i, path in enumerate(val_data):
        name = os.path.basename(path)
        print(path)
        im, points = generate_data(path)
        im_save_path = os.path.join(val_save_dir, name)
        im.save(im_save_path)
        gd_save_path = im_save_path.replace('jpg', 'npy')
        np.save(gd_save_path, points)