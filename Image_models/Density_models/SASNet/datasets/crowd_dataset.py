import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import glob
from time import sleep

class CrowdDataset(Dataset):
    def __init__(self, root_path, transform=None):
        root = glob.glob(os.path.join(root_path, 'test_data/images/*.jpg'))
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
       
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img, target = load_data(img_path)
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img) # [3, 350, 1024]
        target = torch.Tensor(target) # [350, 1024]
        return img, target

def load_data(img_path):
    gt_path = img_path.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    while True:
        try:
            gt_file = h5py.File(gt_path)
            break
        except:
            sleep(2)
    target = np.asarray(gt_file['density'])
    return img, target