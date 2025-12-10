import warnings
warnings.filterwarnings('ignore')
import h5py
import PIL.Image as Image
import numpy as np
import argparse
import cv2
import os
import torch
import math

def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)
    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data
    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255
    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0
    return distance_map

def main(args):
    if not os.path.exists(os.path.join(args.input_dir, 'gt_fidt_map_2048')):
        os.makedirs(os.path.join(args.input_dir, 'gt_fidt_map_2048'))
    if not os.path.exists(os.path.join(args.input_dir, 'gt_show_fidt')):
        os.makedirs(os.path.join(args.input_dir, 'gt_show_fidt'))
    f = open(os.path.join(args.input_dir, 'NWPU_list/train.txt'), 'r')
    train_list = f.readlines()
    f = open(os.path.join(args.input_dir, 'NWPU_list/val.txt'), 'r')
    val_list = f.readlines()
    # process train part
    for i in range(len(train_list)):
        fname = train_list[i].split(' ')[0] + '.jpg'
        img_path = args.input_dir + '/images_2048/' + fname  # 2048 for train
        img = cv2.imread(img_path)  # [1416, 2048, 3]
        Img_data_pil = Image.open(img_path).convert('RGB')
        k = np.zeros((img.shape[0], img.shape[1]))  # [1416, 2048]
        mat_path = img_path.replace('images', 'gt_npydata').replace('jpg', 'npy')
        with open(mat_path, 'rb') as outfile:
            gt = np.load(outfile).tolist()  # [46, 2]
        fidt_map = fidt_generate1(np.array(Img_data_pil), gt, 1)
        gt_show_path = img_path.replace('images_2048', 'gt_show_fidt')
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        kpoint = k.copy()
        kpoint = kpoint.astype(np.uint8)  # [1416, 2048]
        with h5py.File(img_path.replace('images_2048', 'gt_fidt_map_2048').replace('jpg', 'h5'), 'w') as hf:
            hf['kpoint'] = kpoint
            hf['fidt_map'] = fidt_map
        fidt_map = fidt_map / np.max(fidt_map) * 255
        fidt_map = fidt_map.astype(np.uint8)
        fidt_map = cv2.applyColorMap(fidt_map, 2)
        result = fidt_map
        cv2.imwrite(gt_show_path, result)
    # process val part
    for i in range(len(val_list)):
        fname = val_list[i].split(' ')[0] + '.jpg'
        img_path = args.input_dir + '/images/' + fname
        img = cv2.imread(img_path)
        image_s = cv2.imread(img_path.replace('images', 'images_2048'))
        Img_data_pil = Image.open(img_path).convert('RGB')
        if img.shape[1] >= img.shape[0] and img.shape[1] >= 2048:
            rate1 = 2048.0 / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1, interpolation=cv2.INTER_CUBIC)
            Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)
        elif img.shape[0] >= img.shape[1] and img.shape[0] >= 2048:
            rate1 = 2048.0 / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1, interpolation=cv2.INTER_CUBIC)
            Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)
        rate = img.shape[0] / image_s.shape[0]
        point_map = np.zeros((img.shape[0], img.shape[1]))
        mat_path = img_path.replace('images', 'gt_npydata_2048').replace('jpg', 'npy')
        with open(mat_path, 'rb') as outfile:
            gt = (np.load(outfile) * rate).tolist()
        fidt_map = fidt_generate1(np.array(Img_data_pil), gt, 1)
        gt_show_path = img_path.replace('images', 'gt_show_fidt')
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                point_map[int(gt[i][1]), int(gt[i][0])] = 1
        kpoint = point_map.copy()
        kpoint = kpoint.astype(np.uint8)
        with h5py.File(img_path.replace('images', 'gt_fidt_map_2048').replace('jpg', 'h5'), 'w') as hf:
            hf['kpoint'] = kpoint
            hf['fidt_map'] = fidt_map
        fidt_map = fidt_map / np.max(fidt_map) * 255
        fidt_map = fidt_map.astype(np.uint8)
        fidt_map = cv2.applyColorMap(fidt_map, 2)
        result = fidt_map
        cv2.imwrite(gt_show_path, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='NWPU-Crowd')
    parser.add_argument('--input_dir', type=str, default='datasets/NWPU_CLTR')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)