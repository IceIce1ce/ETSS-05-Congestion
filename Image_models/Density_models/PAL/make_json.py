import json
import os
import argparse

def get_train(args):
    if args.type_dataset == 'sha' or args.type_dataset == 'shb':
        path = os.path.join(args.input_dir, 'train_data', 'images')
        filenames = os.listdir(path)
        pathname = [os.path.join(path, filename) for filename in filenames]
        with open(args.train_json, 'w') as f:
            json.dump(pathname, f)
    else:
        print('This dataset does not exist')
        raise NotImplementedError

def get_test(args):
    if args.type_dataset == 'sha' or args.type_dataset == 'shb':
        path = os.path.join(args.input_dir, 'test_data', 'images')
        filenames = os.listdir(path)
        pathname = [os.path.join(path, filename) for filename in filenames]
        with open(args.val_json, 'w') as f:
            json.dump(pathname, f)
    else:
        print('This dataset does not exist')
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A')
    parser.add_argument('--train_json', type=str, default='A_train.json')
    parser.add_argument('--val_json', type=str, default='A_val.json')
    args = parser.parse_args()

    get_train(args)
    get_test(args)