import json
from os.path import join
import os
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='datasets/FDST')
    parser.add_argument('--output_dir', type=str, default='datasets')
    args = parser.parse_args()

    train_folders = join(args.input_dir, 'train_data')
    test_folders = join(args.input_dir, 'test_data')
    output_train_all = join(args.output_dir, 'train_all.json')
    output_train = join(args.output_dir, 'train.json')
    output_val = join(args.output_dir, 'val.json')
    output_test = join(args.output_dir, 'test.json')
    train_all_img_list = []
    test_img_list = []
    for root,dirs, files in os.walk(train_folders):
        for file_name in files:
            if file_name.endswith('.jpg'):
                train_all_img_list.append(join(root,file_name))
    for root,dirs, files in os.walk(test_folders):
        for file_name in files:
            if file_name.endswith('.jpg'):
                test_img_list.append(join(root,file_name))
    all_num = len(train_all_img_list)
    train_num = int(all_num * 0.8)
    random.shuffle(train_all_img_list)
    train_img_list = train_all_img_list[:train_num]
    val_img_list = train_all_img_list[train_num:]
    with open(output_train_all,'w') as f:
        json.dump(train_all_img_list,f)
    with open(output_train,'w') as f:
        json.dump(train_img_list,f)
    with open(output_val,'w') as f:
        json.dump(val_img_list,f)
    with open(output_test,'w') as f:
        json.dump(test_img_list,f)