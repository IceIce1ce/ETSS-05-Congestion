import warnings
warnings.filterwarnings('ignore')
import os
import h5py
import PIL.Image as Image
from PIL import ImageDraw
import numpy as np
import argparse
import cv2

def main(args):
    if not os.path.exists(os.path.join(args.input_dir, 'gt_detr_map')):
        os.makedirs(os.path.join(args.input_dir, 'gt_detr_map'))
    f = open("data/NWPU_list/train.txt", "r")
    train_list = f.readlines()
    f = open("data/NWPU_list/val.txt", "r")
    val_list = f.readlines()
    ### visualize GT ###
    vis_dir = os.path.join(args.input_dir, 'gt_show')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    ### visualize GT ###
    # process train part
    for i in range(len(train_list)):
        fname = train_list[i].split(' ')[0] + '.jpg'
        img_path = os.path.join(args.input_dir, 'images_2048', fname) # 2048 for train
        img = cv2.imread(img_path) # [1416, 2048, 3]
        Img_data_pil = Image.open(img_path).convert('RGB')
        k = np.zeros((img.shape[0], img.shape[1])) # [1416, 2048]
        point_map =  np.zeros((img.shape[0], img.shape[1], 3)) + 255 # [1416, 2048, 3]
        mat_path = img_path.replace('images', 'gt_npydata').replace('jpg', 'npy')
        with open(mat_path, 'rb') as outfile:
            gt = np.load(outfile).tolist() # [46, 2]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
                point_map = cv2.circle(point_map, (int(gt[i][0]), int(gt[i][1])), 5, (0, 0, 0), -1)
        kpoint = k.copy()
        kpoint = kpoint.astype(np.uint8) # [1416, 2048]
        with h5py.File(img_path.replace('images_2048', 'gt_detr_map').replace('jpg', 'h5'), 'w') as hf:
            hf['kpoint'] = kpoint
            hf['image'] = Img_data_pil
        ### visualize GT ###
        vis_img = Img_data_pil.copy()
        draw = ImageDraw.Draw(vis_img)
        coords = np.argwhere(kpoint > 0)
        for (r, c) in coords:
            draw.ellipse((c - 2, r - 2, c + 2, r + 2), fill=(255, 0, 0))
        vis_path = img_path.replace('images_2048', 'gt_show')
        vis_img.save(vis_path)
        ### visualize GT ###
    # process val part
    for i in range(len(val_list)):
        fname = val_list[i].split(' ')[0] + '.jpg'
        img_path = os.path.join(args.input_dir, 'images', fname) # 4096 for val
        img = cv2.imread(img_path)
        image_s = cv2.imread(img_path.replace('images', 'images_2048'))
        Img_data_pil = Image.open(img_path).convert('RGB')
        if img.shape[1] >= img.shape[0] and img.shape[1] >= 4096:
            rate1 = 4096.0 / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1, interpolation=cv2.INTER_CUBIC)
            Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)
        elif img.shape[0] >= img.shape[1] and img.shape[0] >= 4096:
            rate1 = 4096.0 / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1, interpolation=cv2.INTER_CUBIC)
            Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)
        rate = img.shape[0] / image_s.shape[0]
        point_map = np.zeros((img.shape[0], img.shape[1]))
        mat_path = img_path.replace('images', 'gt_npydata_2048').replace('jpg', 'npy')
        with open(mat_path, 'rb') as outfile:
            gt = (np.load(outfile) * rate).tolist()
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                point_map[int(gt[i][1]), int(gt[i][0])] = 1
        kpoint = point_map.copy()
        kpoint = kpoint.astype(np.uint8)
        with h5py.File(img_path.replace('images', 'gt_detr_map').replace('jpg', 'h5'), 'w') as hf:
            hf['kpoint'] = kpoint
            hf['image'] = Img_data_pil
        ### visualize GT ###
        vis_img = Img_data_pil.copy()
        draw = ImageDraw.Draw(vis_img)
        coords = np.argwhere(kpoint > 0)
        for (r, c) in coords:
            draw.ellipse((c - 2, r - 2, c + 2, r + 2), fill=(255, 0, 0))
        vis_path = img_path.replace('images', 'gt_show')
        vis_img.save(vis_path)
        ### visualize GT ###

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='NWPU-Crowd')
    parser.add_argument('--input_dir', type=str, default='data/NWPU_CLTR')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)