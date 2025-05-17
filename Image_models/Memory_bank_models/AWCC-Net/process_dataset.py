from PIL import Image
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
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

def generate_data_jhu(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('images', 'gt').replace('.jpg', '.txt')
    points = []
    with open (mat_path, 'r') as f:
        while True:
            point = f.readline()
            if not point:
                break
            point = point.split(' ')[:-1]
            points.append([float(point[0]), float(point[1])])
    points = np.array(points)
    if len(points>0):
        idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
        points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def run_jhu(origin_dir, save_dir, min_size, max_size):
    for phase in ['train', 'val', 'test']:
            sub_dir = os.path.join(origin_dir, phase)
            sub_save_dir = os.path.join(save_dir, phase)
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            im_list = glob(os.path.join(os.path.join(sub_dir, 'images'), '*jpg'))
            for im_path in tqdm(im_list):
                name = os.path.basename(im_path)
                im, points = generate_data_jhu(im_path, min_size, max_size)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path, quality=95)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="datasets/jhu_crowd_v2.0", type=str)
    parser.add_argument('--output_dir', default="datasets/jhu", type=str)
    parser.add_argument('--min_size', type=int, default=512)
    parser.add_argument('--max_size', type=int, default=2048)
    args = parser.parse_args()
    run_jhu(args.input_dir, args.output_dir, args.min_size, args.max_size)