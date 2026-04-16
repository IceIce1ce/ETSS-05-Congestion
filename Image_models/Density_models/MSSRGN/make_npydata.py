import os
import numpy as np
import argparse

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    try:
        sr_train_path = os.path.join(args.input_dir, 'train/images')
        sr_test_path = os.path.join(args.input_dir, 'test/images')
        train_list = []
        for filename in os.listdir(sr_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(os.path.join(sr_train_path, filename))
        train_list.sort()
        np.save(os.path.join(args.output_dir, 'crowdsr_train.npy'), train_list)
        test_list = []
        for filename in os.listdir(sr_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(os.path.join(sr_test_path, filename))
        test_list.sort()
        np.save(os.path.join(args.output_dir, 'crowdsr_test.npy'), test_list)
    except:
        print("The SR-Crowd dataset path is wrong")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SR-Crowd')
    parser.add_argument('--input_dir', type=str, default='datasets/Crowd-SR')
    parser.add_argument('--output_dir', type=str, default='npy_data')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)