import warnings
warnings.filterwarnings('ignore')
import glob
import os
import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
import argparse

def main(args):
    train = os.path.join(args.input_dir, 'train/images/')
    val = os.path.join(args.input_dir, 'val/images/')
    test = os.path.join(args.input_dir, 'test/images/')
    ### visualize GT ###
    for path in [train, val, test]:
        vis_dir = path.replace('images', 'gt_show')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
    ### visualize GT ###
    if not os.path.exists(train.replace('images', 'images_2048')):
        os.makedirs(train.replace('images', 'images_2048'))
    if not os.path.exists(train.replace('images', 'gt_detr_map_2048')):
        os.makedirs(train.replace('images', 'gt_detr_map_2048'))
    if not os.path.exists(val.replace('images', 'images_2048')):
        os.makedirs(val.replace('images', 'images_2048'))
    if not os.path.exists(val.replace('images', 'gt_detr_map_2048')):
        os.makedirs(val.replace('images', 'gt_detr_map_2048'))
    if not os.path.exists(test.replace('images', 'images_2048')):
        os.makedirs(test.replace('images', 'images_2048'))
    if not os.path.exists(test.replace('images', 'gt_detr_map_2048')):
        os.makedirs(test.replace('images', 'gt_detr_map_2048'))
    path_sets = [train, test, val]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    img_paths.sort()
    for img_path in img_paths:
        img = cv2.imread(img_path) # [955, 1300, 3]
        Img_data_pil = Image.open(img_path).convert('RGB')
        rate1 = 1
        rate2 = 1
        if img.shape[1] >= img.shape[0] and img.shape[1] >= 2048:
            rate1 = 2048.0 / img.shape[1]
        elif img.shape[0] >= img.shape[1] and img.shape[0] >= 2048:
            rate1 = 2048.0 / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1, interpolation=cv2.INTER_CUBIC) # [955, 1300, 3]
        Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]) ,Image.ANTIALIAS)
        min_shape = 512.0
        if img.shape[1] <= img.shape[0] and img.shape[1] <= min_shape:
            rate2 = min_shape / img.shape[1]
        elif img.shape[0] <= img.shape[1] and img.shape[0] <= min_shape:
            rate2 = min_shape / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=rate2, fy=rate2, interpolation=cv2.INTER_CUBIC) # [955, 1300, 3]
        Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)
        rate = rate1 * rate2
        k = np.zeros((img.shape[0], img.shape[1])) # [955, 1300]
        gt_file = np.loadtxt(img_path.replace('images', 'gt').replace('jpg', 'txt')) # [43, 6]
        try:
            y = gt_file[:, 0] * rate
            x = gt_file[:, 1] * rate
            for i in range(0, len(x)):
                if int(x[i]) < img.shape[0] and int(y[i]) < img.shape[1]:
                    k[int(x[i]), int(y[i])] += 1
        except Exception:
            try:
                y = gt_file[0] * rate
                x = gt_file[1] * rate
                for i in range(0, 1):
                    if int(x) < img.shape[0] and int(y) < img.shape[1]:
                        k[int(x), int(y)] += 1
            except Exception:
                k = np.zeros((img.shape[0], img.shape[1]))
        kpoint = k.copy()
        kpoint = kpoint.astype(np.uint8) # [955, 1300]
        with h5py.File(img_path.replace('images', 'gt_detr_map_2048').replace('jpg', 'h5'), 'w') as hf:
            hf['kpoint'] = kpoint
            hf['image'] = Img_data_pil
        cv2.imwrite(img_path.replace('images', 'images_2048'), img)
        ### visualize GT ###
        vis_img = Img_data_pil.copy()
        draw = ImageDraw.Draw(vis_img)
        coords = np.argwhere(kpoint > 0)
        for (r, c) in coords:
            draw.ellipse((c - 2, r - 2, c + 2, r + 2), fill=(255, 0, 0))
        vis_path = img_path.replace('images', 'gt_show')
        vis_img.save(vis_path)
        ### visualize GT ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='JHU-Crowd')
    parser.add_argument('--input_dir', type=str, default='data/jhu_crowd_v2.0')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)