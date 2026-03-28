import numpy as np
import os
import torch
from torch.utils import data
from PIL import Image
if __name__ == '__main__':
    from utils import NormalSample
else:
    from .utils import NormalSample

class SHHA(data.Dataset):
    def __init__(self, root_path, mode):
        self.imgids = []
        imtype = 'jpg'
        for imgf in os.listdir(os.path.join(root_path, mode + '_data', 'images')):
            if imtype in imgf:
                self.imgids.append(imgf.replace(f'.{imtype}', ''))
        self.imgpath = os.path.join(root_path, mode + '_data', 'images', '{}' + f'.{imtype}')
        self.dotpath = os.path.join(root_path, mode + '_data', 'new-anno', 'GT_{}.npy')
        self.normalfunc = NormalSample(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=(mode=='train'))
    
    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, index):
        smpid = self.imgids[index]
        img, dotseq = self.readSampleFromId(smpid, resize_factor=1)
        image, dotseq = self.normalfunc(img, dotseq)
        return image, dotseq, smpid

    def readSampleFromId(self, smpid, resize_factor=1):
        imgpath = self.imgpath.format(smpid)
        img = Image.open(imgpath).convert('RGB')
        if resize_factor > 1:
            img = img.resize((img.width*resize_factor, img.height*resize_factor), Image.LANCZOS)
        img = self.normalfunc.im2tensor(img)
        dotseq = torch.from_numpy(np.load(self.dotpath.format(smpid)))[:, :2] * resize_factor
        return img, dotseq

    @staticmethod
    def collate_fn(samples):
        images, seqinfo, imgnames = zip(*samples)
        images = torch.cat(images, dim=0)
        seqinfo = sum(seqinfo, [])
        return images, seqinfo, imgnames