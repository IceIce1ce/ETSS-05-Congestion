import os
import glob
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torchvision import transforms

class Countmap_Dataset(Dataset):
    def __init__(self, img_dir,tar_dir, rgb_dir,transform=None,if_test = False, IF_loadmem=False):
        self.IF_loadmem = IF_loadmem
        self.IF_loadFinished = False
        self.image_mem = []
        self.target_mem = []
        self.img_dir = img_dir
        self.tar_dir = tar_dir
        self.transform = transform
        mat = sio.loadmat(rgb_dir)
        self.rgb = mat['rgbMean'].reshape(1,1,3) 
        img_name = os.path.join(self.img_dir,'*.jpg')
        self.filelist =  glob.glob(img_name)
        self.dataset_len = len(self.filelist)
        self.if_test = if_test
        self.DIV = 64

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if (not self.IF_loadmem) or (not self.IF_loadFinished):
            img_name = self.filelist[idx]
            image = Image.open(img_name).convert('RGB')
            image = transforms.ToTensor()(image)
            image = get_pad(image,DIV=64)
            image = image - torch.Tensor(self.rgb).view(3,1,1)           
            (filepath,tempfilename) = os.path.split(img_name)
            (name,extension) = os.path.splitext(tempfilename)
            mat_dir = os.path.join( self.tar_dir, '%s.mat' % (name) )
            mat = sio.loadmat(mat_dir)
            if self.IF_loadmem:
                self.image_mem.append(image)
                self.target_mem.append(mat)
                if len(self.image_mem) == self.dataset_len:
                    self.IF_loadFinished = True
        else:
            image = self.image_mem[idx]
            mat = self.target_mem[idx]
        all_num = mat['dot_num'].reshape((1,1))
        sample = {'image': image,'all_num':all_num}
        sample['all_num'] = torch.from_numpy(sample['all_num'])
        if self.transform:
            for t in self.transform:
                sample = t(sample)
        sample['name'] = name
        return sample

def get_pad(inputs,DIV=64):
    h,w = inputs.size()[-2:]
    ph,pw = (DIV-h%DIV),(DIV-w%DIV)
    tmp_pad = [0,0,0,0]
    if (ph!=DIV): 
        tmp_pad[2],tmp_pad[3] = 0,ph
    if (pw!=DIV):
        tmp_pad[0],tmp_pad[1] = 0,pw
    inputs = F.pad(inputs,tmp_pad)
    return inputs