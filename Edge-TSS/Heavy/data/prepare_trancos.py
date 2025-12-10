import warnings
warnings.filterwarnings("ignore")
import glob
import os
import cv2
import h5py
import numpy as np
import scipy.io as io
import argparse
from PIL import Image, ImageDraw

def main(args):
    part_train = os.path.join(args.input_dir, 'train_data', 'images')
    part_test = os.path.join(args.input_dir, 'test_data', 'images')
    path_sets = [part_train, part_test]
    if not os.path.exists(part_train.replace('images', 'gt_detr_map')):
        os.makedirs(part_train.replace('images', 'gt_detr_map'))
    if not os.path.exists(part_test.replace('images', 'gt_detr_map')):
        os.makedirs(part_test.replace('images', 'gt_detr_map'))
    if not os.path.exists(part_train.replace('images', 'gt_show')):
        os.makedirs(part_train.replace('images', 'gt_show'))
    if not os.path.exists(part_test.replace('images', 'gt_show')):
        os.makedirs(part_test.replace('images', 'gt_show'))
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    img_paths.sort()
    for img_path in img_paths:
        img = cv2.imread(img_path)
        Img_data_pil = Image.open(img_path).convert('RGB')
        k = np.zeros((img.shape[0], img.shape[1]))
        mat_path = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth'))
        gt = mat_path["image_info"][0][0][0][0][0].tolist()
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        kpoint = k.copy()
        kpoint = kpoint.astype(np.uint8)
        with h5py.File(img_path.replace('images', 'gt_detr_map').replace('jpg', 'h5'), 'w') as hf:
            hf['kpoint'] = kpoint
            hf['image'] = Img_data_pil
        vis_img = Img_data_pil.copy()
        draw = ImageDraw.Draw(vis_img)
        coords = np.argwhere(kpoint > 0)
        for (r, c) in coords:
            draw.ellipse((c - 2, r - 2, c + 2, r + 2), fill=(255, 0, 0))
        vis_path = img_path.replace('images', 'gt_show')
        vis_img.save(vis_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='TRANCOS')
    parser.add_argument('--input_dir', type=str, default='datasets/TRANCOS')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)