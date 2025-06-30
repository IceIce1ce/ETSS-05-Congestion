from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, main_transform=None, img_transform=None, gt_transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root + '/img', fname)
        img = Image.open(fpath).convert('RGB') # [1280, 1920, 3]
        den = pd.read_csv(os.path.join(self.root + '/den', os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values
        den = den.astype(np.float32, copy=False) # [1280, 1920]
        den = Image.fromarray(den)
        if self.main_transform is not None:
            img, den = self.main_transform(img,den)
        if self.img_transform is not None:
            img = self.img_transform(img) # [3, 1280, 1920]
        if self.gt_transform is not None:
            den = self.gt_transform(den) # [1280, 1920]
        return img, den