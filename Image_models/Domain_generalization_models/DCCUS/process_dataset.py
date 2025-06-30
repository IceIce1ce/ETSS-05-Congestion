import argparse
import os
import cv2
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter

def get_density_map_gaussian(im, points, k_size, sigma):
    h, w = im.shape[:2]
    im_density = np.zeros((h, w)) # [2032, 1584]
    if len(points) == 0:
        return im_density
    for j in range(len(points)):
        x = min(w, max(1, abs(int(np.floor(points[j, 0])))))
        y = min(h, max(1, abs(int(np.floor(points[j, 1])))))
        if x > w or y > h:
            continue
        x1 = x - int(np.floor(k_size / 2))
        y1 = y - int(np.floor(k_size / 2))
        x2 = x + int(np.floor(k_size / 2))
        y2 = y + int(np.floor(k_size / 2))
        dfx1, dfy1, dfx2, dfy2 = 0, 0, 0, 0
        change_H = False
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dfy1 = abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = k_size - dfx2
        y2h = k_size - dfy2
        if change_H:
            H = np.zeros((y2h - y1h + 1, x2h - x1h + 1))
            H[int(H.shape[0] / 2), int(H.shape[1] / 2)] = 1
            H = gaussian_filter(H, sigma=sigma, mode='constant')
            H = H / np.sum(H) if np.sum(H) != 0 else H
        else:
            H = np.zeros((k_size, k_size))
            H[int(k_size / 2), int(k_size / 2)] = 1
            H = gaussian_filter(H, sigma=sigma, mode='constant')
            H = H / np.sum(H) if np.sum(H) != 0 else H
        im_density[y1 - 1:y2, x1 - 1:x2] += H
    return im_density

def get_dot_map(im, points):
    h, w = im.shape[:2]
    dot_map = np.zeros((h, w)) # [2032, 1584]
    if len(points) == 0:
        return dot_map
    points = points.astype(np.int32) # [214, 2]
    for i_dot in range(len(points)):
        x = points[i_dot, 0] # [1]
        y = points[i_dot, 1] # [1]
        if x > w or y > h or x <= 0 or y <= 0:
            continue
        dot_map[y - 1, x - 1] += 1
    return dot_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech')
    parser.add_argument('--output_dir', type=str, default='data/sha')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    for split in ['train_data', 'test_data']:
        init_max_size = [2048, 2048]
        min_size = [576, 768]
        save_type = [1, 1, 1, 1] # img, density, dot, vis
        split_path = os.path.join(args.output_dir, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        path_img = os.path.join(split_path, 'img')
        path_den = os.path.join(split_path, 'den')
        path_dot = os.path.join(split_path, 'dot')
        path_vis = os.path.join(split_path, 'vis')
        os.makedirs(path_img, exist_ok=True)
        os.makedirs(path_den, exist_ok=True)
        os.makedirs(path_dot, exist_ok=True)
        os.makedirs(path_vis, exist_ok=True)
        img_list = [f for f in os.listdir(os.path.join(args.input_dir, split, 'images')) if f.endswith('.jpg')]
        for img_name in img_list:
            name_split = img_name.split('.jpg')
            mat_name = 'GT_' + name_split[0] + '.mat'
            im = cv2.imread(os.path.join(args.input_dir, split, 'images', img_name)) # [640, 480, 3]
            h, w, c = im.shape
            rate = init_max_size[0] / h
            rate_w = w * rate
            if rate_w > init_max_size[1]:
                rate = init_max_size[1] / w
            tmp_h = int(h * rate / 16) * 16
            if tmp_h < min_size[0]:
                rate = min_size[0] / h
            tmp_w = int(w * rate / 16) * 16
            if tmp_w < min_size[1]:
                rate = min_size[1] / w
            tmp_h = int(h * rate / 16) * 16
            tmp_w = int(w * rate / 16) * 16
            rate_h = tmp_h / h
            rate_w = tmp_w / w
            im = cv2.resize(im, (tmp_w, tmp_h)) # [2048, 1536, 3]
            if save_type[0] == 1: # save original imgs
                cv2.imwrite(os.path.join(path_img, img_name), im)
            mat_path = os.path.join(args.input_dir, split, 'ground-truth', mat_name)
            if not os.path.exists(mat_path):
                continue
            mat_data = scipy.io.loadmat(mat_path)
            ann_points = mat_data['image_info'][0][0][0][0][0] # [77, 2]
            if ann_points.size > 0:
                ann_points[:, 0] = ann_points[:, 0] * rate_w
                ann_points[:, 1] = ann_points[:, 1] * rate_h
                check_list = np.ones(ann_points.shape[0], dtype=bool) # [77]
                for j in range(ann_points.shape[0]):
                    x = int(np.ceil(ann_points[j, 0]))
                    y = int(np.ceil(ann_points[j, 1]))
                    if x > tmp_w or y > tmp_h or x <= 0 or y <= 0:
                        check_list[j] = False
                ann_points = ann_points[check_list, :] # [77, 2]
            if save_type[1] == 1: # save density maps
                im_density = get_density_map_gaussian(im, ann_points, 15, 4) # [2048, 1536]
                np.savetxt(os.path.join(path_den, name_split[0] + '.csv'), im_density, delimiter=',')
            if save_type[2] == 1: # save dot maps
                im_dots = get_dot_map(im, ann_points) # [2048, 1536]
                im_dots = (im_dots * 255).astype(np.uint8) # [2048, 1536]
                cv2.imwrite(os.path.join(path_dot, name_split[0] + '.png'), im_dots)
            if save_type[3] == 1: # save vis
                im_rgb = im.copy() # [2048, 1536, 3]
                if ann_points.size > 0:
                    for point in ann_points:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(im_rgb, (x, y), 5, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(path_vis, name_split[0] + '.jpg'), im_rgb)