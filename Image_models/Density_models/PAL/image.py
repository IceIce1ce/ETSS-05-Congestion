import random
from PIL import Image
import numpy as np
import h5py
import cv2
import os

def load_data_masked_data(img_path, train=True, args=None):
    if train:
        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth')
        if args.type_dataset == 'sha':
            mask_ratio = os.path.join('datasets/masked_data/part_A/train_data', args.gt_ratio, 'mask')
        elif args.type_dataset == 'shb':
            mask_ratio = os.path.join('datasets/masked_data/part_B/train_data', args.gt_ratio, 'mask')
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        mask_path = os.path.join(mask_ratio, img_path.replace('IMG','GT_IMG').replace('.jpg','.png').split('/')[-1])
    else:
        gt_path = img_path.replace('.jpg','.h5').replace('images','ground-truth')
        mask_path = img_path
    img = Image.open(img_path).convert('RGB') # [625, 1024, 3]
    gt_file = h5py.File(gt_path,'r')
    target = np.asarray(gt_file['density']) # [625, 1024]
    mask = Image.open(mask_path).convert('L') # [645, 1024]
    target_20 = np.asarray(gt_file['density']) # [645, 1024]
    if train:
        while 1:
            ratio = 0.5
            crop_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            rdn_value = random.random()
            if rdn_value < 0.25:
                dx = 0
                dy = 0
            elif rdn_value < 0.5:
                dx = int(img.size[0] * ratio)
                dy = 0
            elif rdn_value < 0.75:
                dx = 0
                dy = int(img.size[1] * ratio)
            else:
                dx = int(img.size[0] * ratio)
                dy = int(img.size[1] * ratio)
            if args.use_random_crop:
                dx = random.randint(0,int(img.size[0] * ratio))
                dy = random.randint(0,int(img.size[1] * ratio))
            img_rt = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target_rt = target[dy:(crop_size[1] + dy), dx:(crop_size[0] + dx)]
            target_20_rt = target_20[dy:(crop_size[1] + dy), dx:(crop_size[0] + dx)]
            mask_rt = mask.crop((dx, dy,crop_size[0] + dx,crop_size[1] + dy))
            if random.random() > 0.8:
                target_rt = np.fliplr(target_rt)
                target_20_rt = np.fliplr(target_20_rt)
                mask_rt = mask_rt.transpose(Image.FLIP_LEFT_RIGHT)
                img_rt = img_rt.transpose(Image.FLIP_LEFT_RIGHT)
            if args.type_dataset == 'sha':
                break_cnt = 1500 * 255
            elif args.type_dataset == 'shb':
                break_cnt = 3000 * 255
            else:
                print('This dataset does not exist')
                raise NotImplementedError
            if np.array(mask_rt).sum() > break_cnt:
                break
        target_rt = cv2.resize(target_rt, (int(target_rt.shape[1] / 8), int(target_rt.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64 # [48, 64]
        target_20_rt = cv2.resize(target_20_rt, (int(target_20_rt.shape[1] / 8), int(target_20_rt.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64 # [48, 64]
    if train:
        mask_rt = np.array(mask_rt) # [193, 290]
        mask_rt = cv2.resize(mask_rt, (int(mask_rt.shape[0] / 8), int(mask_rt.shape[1] / 8)), interpolation=cv2.INTER_CUBIC) # [36, 24]
        return img_rt, target_rt, mask_rt, target_20_rt
    else:
        target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64 # [95, 128]
        if args.is_eval:
            return img, target, target, target, img_path
        else:
            return img, target, target, target