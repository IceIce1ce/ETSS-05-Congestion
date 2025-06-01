import h5py
import numpy as np
import os
import argparse
import cv2

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def generate_density_map(shape=(5, 5), points=None, f_sz=15, sigma=4):
    im_density = np.zeros(shape[0:2])
    h, w = shape[0:2]
    if len(points) == 0:
        return im_density
    for j in range(len(points)):
        H = matlab_style_gauss2D((f_sz, f_sz), sigma)
        x = np.minimum(w, np.maximum(1, np.abs(np.int32(np.floor(points[j, 0])))))
        y = np.minimum(h, np.maximum(1, np.abs(np.int32(np.floor(points[j, 1])))))
        if x > w or y > h:
            continue
        x1 = x - np.int32(np.floor(f_sz / 2))
        y1 = y - np.int32(np.floor(f_sz / 2))
        x2 = x + np.int32(np.floor(f_sz / 2))
        y2 = y + np.int32(np.floor(f_sz / 2))
        dx1 = 0
        dy1 = 0
        dx2 = 0
        dy2 = 0
        change_H = False
        if x1 < 1:
            dx1 = np.abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dy1 = np.abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dx1
        y1h = 1 + dy1
        x2h = f_sz - dx2
        y2h = f_sz - dy2
        if change_H: 
            H = matlab_style_gauss2D((y2h - y1h + 1, x2h - x1h + 1), sigma)
        im_density[y1 - 1:y2, x1 - 1:x2] = im_density[y1 - 1:y2, x1 - 1:x2] + H
    return im_density

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A_final')
    args = parser.parse_args()

    print('Process dataset:', args.input_dir.split('/')[1])
    for split in ['train_data', 'test_data']:
        img_paths = []
        for root, dirs, files in os.walk(os.path.join(args.input_dir, split, 'images')):
            for img_path in files:
                if img_path.endswith('.jpg'):
                    img_paths.append(os.path.join(root, img_path))
        for img_path in img_paths:
            gt_path = img_path.replace('.jpg', '.txt')
            gt = []
            with open(gt_path) as f_label:
                for line in f_label:
                    x = float(line.strip().split(' ')[0])
                    y = float(line.strip().split(' ')[1])
                    gt.append([x, y])
            image = cv2.imread(img_path) # [685, 1024, 3]
            positions = generate_density_map(shape=image.shape, points=np.array(gt), f_sz=15, sigma=4) # [685, 1024]
            with h5py.File(img_path.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth'), 'w') as hf:
                hf['density'] = positions