import json
from os.path import join
import os
import random
import argparse

def is_valid_image_name(filename):
    name_without_ext = filename[:-4]
    return name_without_ext.isdigit()

def main(args):
    train_folders = join(args.input_dir, 'train_data')
    test_folders = join(args.input_dir, 'test_data')
    output_train_all = join(args.input_dir, 'train_all.json')
    output_train = join(args.input_dir, 'train.json')
    output_val = join(args.input_dir, 'val.json')
    output_test = join(args.input_dir, 'test.json')
    random.seed(42)
    train_img_list = []
    val_img_list = []
    test_img_list = []
    dirs = next(os.walk(train_folders))[1]
    train_dirs = random.sample(dirs, int(len(dirs) * 0.8))
    for dir_name in train_dirs:
        path = join(train_folders, dir_name)
        for _, _, files in os.walk(path):
            for file_name in files:
                if file_name.endswith('.jpg') and is_valid_image_name(file_name):
                    train_img_list.append(join(path, file_name))
    val_dirs = list(set(dirs).difference(train_dirs))
    for dir_name in val_dirs:
        path = join(train_folders, dir_name)
        for _, _, files in os.walk(path):
            for file_name in files:
                if file_name.endswith('.jpg') and is_valid_image_name(file_name):
                    val_img_list.append(join(path, file_name))
    for root, dirs, files in os.walk(test_folders):
        for file_name in files:
            if file_name.endswith('.jpg') and is_valid_image_name(file_name):
                test_img_list.append(join(root, file_name))
    random.shuffle(train_img_list)
    with open(output_train_all, 'w') as f:
        json.dump(train_img_list, f)
    with open(output_train, 'w') as f:
        json.dump(train_img_list, f)
    with open(output_val, 'w') as f:
        json.dump(val_img_list, f)
    with open(output_test, 'w') as f:
        json.dump(test_img_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='FDST')
    parser.add_argument('--input_dir', type=str, default='datasets/FDST')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)