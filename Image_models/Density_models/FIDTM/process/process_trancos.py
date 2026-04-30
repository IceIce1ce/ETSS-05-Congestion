import os
import numpy as np
from scipy.io import savemat
import argparse

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

def main(args):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='TRANCOS')
    parser.add_argument('--input_dir', type=str, default='datasets/TRANCOS')
    args = parser.parse_args()

    print('Process dataset', args.type_dataset)
    main(args)