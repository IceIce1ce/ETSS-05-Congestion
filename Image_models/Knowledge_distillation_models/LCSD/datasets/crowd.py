import torch.utils.data as data
import os
import cv2
from torchvision import transforms
import numpy as np
        
class Crowd(data.Dataset):
    def __init__(self, root_path, method, downsample_ratio):
        self.root_path = root_path
        self.downsample_ratio = downsample_ratio
        if method == 'train':
            im_list_file = os.path.join(root_path, "train_data.list")
        elif method == 'test':
            im_list_file = os.path.join(root_path, "test_data.list")
        with open(im_list_file,'r') as f:
            self.im_list = f.read().split('\n')
        if '' in self.im_list:
            self.im_list.remove('')
        if method not in ['train', 'test']:
            raise Exception("not implement")
        self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        pair_path = self.im_list[item]
        img_path = os.path.join(self.root_path, pair_path.split()[0])
        gd_path = os.path.join(self.root_path, pair_path.split()[1])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        keypoints = np.loadtxt(gd_path)
        if len(keypoints.shape) == 1 and keypoints.shape[0] == 2:
            keypoints = np.expand_dims(keypoints,0)
        img_height = img.shape[0]
        img_width = img.shape[1]
        if img.shape[0] % self.downsample_ratio:
            img_height = round(img.shape[0] / self.downsample_ratio) * self.downsample_ratio
        if img.shape[1] % self.downsample_ratio:
            img_width = round(img.shape[1] / self.downsample_ratio) * self.downsample_ratio
        if len(keypoints) > 0:
            keypoints[:, 0] = keypoints[:, 0] * (img_width/img.shape[1])
            keypoints[:, 1] = keypoints[:, 1] * (img_height/img.shape[0])
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        img_tensor = self.trans(img)
        base_name = os.path.basename(img_path)
        name = base_name.split('.')[0]
        return img, img_tensor, keypoints, name