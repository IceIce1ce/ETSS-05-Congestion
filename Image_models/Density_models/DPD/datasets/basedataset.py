import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from misc.transforms import RandomAugment
import json

class Dataset(data.Dataset):
    def __init__(self, datasetname, mode, **argv):
        self.mode = mode
        self.datasetname = datasetname
        self.img_path = []
        self.json_path = []
        self.mask_path = []
        self.box_gt = []
        self.info = []
        for data_infor in argv['list_file']:
            data_path, imgId_txt, box_gt_txt = data_infor['data_path'], data_infor['imgId_txt'], data_infor['box_gt_txt']
            with open(os.path.join(data_path, imgId_txt)) as f:
                lines = f.readlines()
            box_gt_Info = []
            if self.mode == 'val':
                box_gt_Info= self.read_box_gt(os.path.join(data_path, box_gt_txt))
            if "NWPU" in data_path:
                for line in lines:
                    splited = line.strip().split()
                    self.img_path.append(os.path.join(data_path, 'images', splited[0] + '.jpg'))
                    self.mask_path.append(os.path.join(data_path, 'mask_50_60', splited[0] + '.png'))
                    if self.mode == 'val':
                        self.box_gt.append(box_gt_Info[int(splited[0])])
                    self.info.append(splited[1:3])
            elif "QNRF" in data_path or "JHU" in data_path:
                for line in lines:
                    line = line.strip()
                    self.img_path.append(os.path.join(data_path, 'images', line + '.jpg'))
                    self.mask_path.append(os.path.join(data_path, 'masks', line + '.png'))
                    if self.mode == 'val':
                        self.box_gt.append(box_gt_Info[int(line)])
            else:
                for line in lines:
                    line = line.strip()
                    self.img_path.append(os.path.join(data_path, 'images', line + '.jpg'))
                    self.mask_path.append(os.path.join(data_path, 'masks', line + '.png'))
                    if self.mode == 'val':
                        self.box_gt.append(box_gt_Info[int(line)])
        self.num_samples = len(self.img_path)
        self.main_transform = None
        if 'main_transform' in argv.keys():
            self.main_transform = argv['main_transform']
        self.img_transform = None
        if 'img_transform' in argv.keys():
            self.img_transform = argv['img_transform']
            self.strong_transform = RandomAugment(argv['args'])
        self.mask_transform = None
        if 'mask_transform' in argv.keys():
            self.mask_transform = argv['mask_transform']
        if self.mode is 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} train images')
        if self.mode is 'val':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} validation images')
        if self.mode is 'test':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} test images')
            with open(data_path + '/res.json', 'r') as f:
                self.res = json.load(f)
    
    def __getitem__(self, index):
        img, mask_map = self.read_image_and_gt(index)
        if self.main_transform is not None:
            img, mask_map  = self.main_transform(img, mask_map)
        if self.img_transform is not None:
            strong_img = self.strong_transform(img) # [1536, 2048, 3]
            img = self.img_transform(img)
            strong_img = self.img_transform(strong_img)
        if self.mask_transform is not None:
            mask_map = self.mask_transform(mask_map)
        if self.mode == 'train':
            return img, strong_img, mask_map
        else:
            return img, mask_map, self.box_gt[index]

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,index):
        img_path = self.img_path[index]
        mask_path = self.mask_path[index]
        img = Image.open(img_path)
        if img.mode is not 'RGB':
            img=img.convert('RGB')
        mask_map =  Image.open(mask_path)  
        return img, mask_map

    def read_box_gt(self,box_gt_file):
        gt_data = {}
        with open(box_gt_file) as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                line_data = [int(i) for i in line]
                idx, num = [line_data[0], line_data[1]]
                if num > 0:
                    points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                    gt_data[idx] = {'num': num, 'points': points_r[:, 0:2], 'sigma': points_r[:, 2:4], 'level': points_r[:, 4]}
                else:
                    gt_data[idx] = {'num': 0, 'points': [], 'sigma': [], 'level': []}
        return gt_data