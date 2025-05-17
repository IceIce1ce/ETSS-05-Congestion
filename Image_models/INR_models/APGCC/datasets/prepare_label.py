# https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/issues/8
from scipy.io import loadmat
import os
import argparse
from tqdm import tqdm
import json

def get_points(mat_path):
    m = loadmat(mat_path)
    return m['image_info'][0][0][0][0][0]

def get_image_list(root_path, sub_path):
    images_path = os.path.join(root_path, sub_path, 'images')
    images = [os.path.join(images_path, im) for im in os.listdir(images_path) if 'jpg' in im]
    return images

def get_gt_from_image(image_path):
    gt_path = os.path.dirname(image_path.replace('images', 'ground-truth'))
    gt_filename = os.path.basename(image_path)
    gt_filename = 'GT_{}'.format(gt_filename.replace('jpg', 'mat'))
    return os.path.join(gt_path, gt_filename)

def ShanghaiTech(root_path, part_name, output_path):
    if part_name not in ['A', 'B']:
        raise NotImplementedError('Supplied dataset part does not exist')
    dataset_splits = ['train_data', 'test_data']
    for split in dataset_splits:
        part_folder = 'part_{}'.format(part_name)
        sub_path = os.path.join(part_folder, '{}'.format(split))
        out_sub_path = os.path.join(part_folder, '{}_data'.format(split))
        images = get_image_list(root_path, sub_path=sub_path)
        try:
            os.makedirs(os.path.join(output_path, out_sub_path))
        except FileExistsError:
            print('Warning, output path already exists, overwriting')
        list_file = []
        for image_path in images:
            gt_path = get_gt_from_image(image_path)
            gt = get_points(gt_path)
            new_labels_file = os.path.join(output_path, out_sub_path, os.path.basename(image_path).replace('jpg', 'txt'))
            with open(new_labels_file, 'w') as fp:
                for p in gt:
                    fp.write('{} {}\n'.format(p[0], p[1]))
            list_file.append((image_path, new_labels_file))
        with open(os.path.join(output_path, part_folder,'{}.list'.format(split)), 'w') as fp:
            for item in list_file:
                fp.write('{} {}\n'.format(item[0], item[1]))

def NWPU(root_path, output_path):
    dataset_splits = ['train.txt', 'val.txt', 'test.txt']
    for split in dataset_splits:
        with open(os.path.join(root_path, split)) as f:
            images = f.read().split('\n')[:-1]
        if split != 'test.txt':
            out_folder = os.path.join(output_path, split.split('.')[0])
            print("DataFolder:", os.path.join(root_path, split), len(images))
            try:
                os.makedirs(out_folder)
            except FileExistsError:
                print('Warning, output path already exists, overwriting')
        list_file = []
        for image_data in tqdm(images):
            iid, l, s = image_data.split(' ')
            img_path = os.path.join(root_path, 'images', iid+'.jpg')
            if split != 'test.txt':
                with open(os.path.join(root_path, 'jsons', iid+'.json')) as f:
                    img_info = json.load(f)
                gt = img_info['points']
                new_labels_file = os.path.join(out_folder, iid+'.txt')
                with open(new_labels_file, 'w') as fp:
                    for p in gt:
                        fp.write('{} {}\n'.format(p[0], p[1]))
                list_file.append((img_path, new_labels_file, l, s))
            else:
                list_file.append((img_path, 'Nan', l, s))
        with open(os.path.join(output_path, '{}.list'.format(split.split('.')[0])), 'w') as fp:
            for item in list_file:
                fp.write('{} {} {} {}\n'.format(item[0], item[1], item[2], item[3]))

def build_datalabel(root_path, dataset, output_path):
    if dataset == 'SHHA':
        ShanghaiTech(root_path, 'A', output_path)
    elif dataset == 'SHHB':
        ShanghaiTech(root_path, 'B', output_path)
    elif dataset == 'NWPU':
        NWPU(root_path, output_path)
    else:
        print('This dataset does not exist')
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_dataset", choices=['SHHA', 'SHHB', 'NWPU'], type=str, default='SHHA')
    parser.add_argument("--dataset_dir", type=str, default='data/ShanghaiTech')
    parser.add_argument("--output_dir", type=str, default='data/sha')
    args = parser.parse_args()

    print('Processing dataset:', args.type_dataset)
    build_datalabel(args.dataset_dir, args.type_dataset, args.output_dir)