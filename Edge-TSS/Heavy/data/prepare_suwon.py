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
from scipy.io import savemat

def convert_txt_to_mat(txt_path, mat_path):
    coordinates = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    coordinates.append([x, y])
    coords_array = np.array(coordinates, dtype=np.float64)
    num_points = len(coordinates)
    location_data = coords_array
    number_data = np.array([[num_points]], dtype=np.uint8)
    structured_array = np.array([[(location_data, number_data)]], dtype=[('location', 'O'), ('number', 'O')])
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = structured_array
    savemat(mat_path, {'image_info': image_info, '__header__': b'MATLAB 5.0 MAT-file, Created by Python', '__version__': '1.0', '__globals__': []})
    return num_points

def process_dataset_folder(data_folder):
    txt_folder = os.path.join(data_folder, 'txt')
    gt_folder = os.path.join(data_folder, 'ground_truth')
    if not os.path.exists(txt_folder):
        print(f"Warning: {txt_folder} does not exist. Skipping...")
        return
    os.makedirs(gt_folder, exist_ok=True)
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    if not txt_files:
        print(f"Warning: No .txt files found in {txt_folder}")
        return
    for txt_file in sorted(txt_files):
        txt_path = os.path.join(txt_folder, txt_file)
        mat_file = txt_file.replace('.txt', '.mat')
        mat_path = os.path.join(gt_folder, mat_file)
        try:
            num_points = convert_txt_to_mat(txt_path, mat_path)
            print(f"✓ {txt_file} -> {mat_file} ({num_points} points)")
        except Exception as e:
            print(f"✗ Error processing {txt_file}: {str(e)}")

def process_mat(args):
    # process train part
    train_folder = os.path.join(args.input_dir, 'train_data')
    if os.path.exists(train_folder):
        process_dataset_folder(train_folder)
    else:
        print(f"Warning: {train_folder} not found")
    # process test part
    test_folder = os.path.join(args.input_dir, 'test_data')
    if os.path.exists(test_folder):
        process_dataset_folder(test_folder)
    else:
        print(f"Warning: {test_folder} not found")

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
    parser.add_argument('--type_dataset', type=str, default='Suwon')
    parser.add_argument('--input_dir', type=str, default='data/Suwon')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    process_mat(args)
    main(args)