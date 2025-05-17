import h5py
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.special import softmax
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--input_dir', default='data/sha', type=str)
    parser.add_argument('--down_sample_rate', type=int, default=8)
    parser.add_argument('--sigma', type=int, default=4)
    parser.add_argument('--use_bg', type=bool, default=True)
    parser.add_argument('--bg_ratio', type=float, default=1.0)
    parser.add_argument('--target_size', type=int, default=256)
    args = parser.parse_args()

    print('Generate GT for dataset:', args.type_dataset)
    building_train = os.path.join(args.input_dir, 'train_data/images')
    gt_path = os.path.join(args.input_dir, 'train_data/bayesian_prior')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    img_paths = []
    for path in [building_train]:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        img = plt.imread(img_path)
        gd_path = img_path.replace('jpg', 'npy')
        gt = np.load(gd_path, allow_pickle=True).astype(np.float32)
        gt = gt[:, :2]
        origin_size = img.shape[0]
        ratio = int(origin_size / args.target_size)
        gt = gt / ratio
        if len(gt) > 0:
            if args.down_sample_rate == 1:
                cood = np.arange(0, args.target_size, step=args.down_sample_rate, dtype=np.float32)
            else:
                cood = np.arange(0, args.target_size, step=args.down_sample_rate, dtype=np.float32) + args.down_sample_rate / 2
            cood = cood[None, :]
            x = gt[:, 0][:, None]
            y = gt[:, 1][:, None]
            x_dis = -2 * np.matmul(x, cood) + x * x + cood * cood
            y_dis = -2 * np.matmul(y, cood) + y * y + cood * cood
            x_dis = np.expand_dims(x_dis, 1)
            y_dis = np.expand_dims(y_dis, 2)
            dis = x_dis + y_dis
            dis = -dis / (2.0 * args.sigma ** 2)
            dis = dis.reshape(len(gt), -1)
            prior_prob = softmax(dis, axis = 1) # [77, 1024]
        else:
            r = args.target_size // args.down_sample_rate
            prior_prob = np.zeros((1, r*r))
        if args.use_bg:
            if args.down_sample_rate == 1:
                cood = np.arange(0, args.target_size, step=args.down_sample_rate, dtype=np.float32)
            else:
                cood = np.arange(0, args.target_size, step=args.down_sample_rate, dtype=np.float32) + args.down_sample_rate / 2
            cood = cood[None, :]
            x = gt[:, 0][:, None]
            y = gt[:, 1][:, None]
            x_dis = -2 * np.matmul(x, cood) + x * x + cood * cood
            y_dis = -2 * np.matmul(y, cood) + y * y + cood * cood
            x_dis = np.expand_dims(x_dis, 1)
            y_dis = np.expand_dims(y_dis, 2)
            dis = x_dis + y_dis
            dis = dis.reshape(len(gt), -1)
            min_dis =np.clip(np.min(dis, axis=0, keepdims=True)[0], a_min=0.0, a_max = None)
            bg_dis = (args.target_size * args.bg_ratio) ** 2 / (min_dis + 1e-5)
            bg_dis = -bg_dis / (2.0 * args.sigma ** 2)
            bg_map = np.exp(bg_dis) / args.sigma / 2.5
            bg_map = bg_map.reshape(1,-1)
            prior_prob = np.concatenate((prior_prob, bg_map), axis = 0) # [78, 1024]
        name = img_path.split('/')
        with h5py.File(os.path.join(gt_path, name[-1][:-4] + '.h5'), 'w') as hf:
            hf['prior_prob'] = prior_prob