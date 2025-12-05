import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import glob
import h5py
import torchvision.transforms as T
import cv2
import random
import json
import scipy.io as sio
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DensityDroneBird(Dataset):
    def __init__(self, data_root, split="train", image_size=256, max_images=None, dens_norm=True, random_flip=True, return_unnorm=False, clip_size=2, stride=1, MEAN=None, STD=None):
        super(DensityDroneBird, self).__init__()
        self.split = split
        self.image_size = image_size
        self.max_images = max_images
        self.dens_norm = dens_norm
        self.random_flip = random_flip
        self.return_unnorm = return_unnorm
        self.clip_size = clip_size
        self.stride = stride
        self.MEAN = MEAN
        self.STD = STD
        with open(os.path.join(data_root, "{}.json".format(self.split)), "r") as f:
            self.image_ids = json.load(f)
        for img_idx in range(len(self.image_ids)):
            self.image_ids[img_idx] = os.path.join(data_root, self.image_ids[img_idx])
        self.image_ids.sort()
        self.seqclips = []
        for img_idx in range(0, len(self.image_ids), self.stride):
            clip = [self.image_ids[img_idx]]
            cur_seq = os.path.basename(self.image_ids[img_idx])[3:6]
            pre_img = self.image_ids[img_idx]
            for i in range(1, self.clip_size):
                seq = os.path.basename(self.image_ids[max(0, img_idx - i)])[3:6]
                if seq == cur_seq:
                    pre_img = self.image_ids[max(0, img_idx - i)]
                clip.append(pre_img)
            self.seqclips.append(clip[::-1])
        if isinstance(self.max_images, int):
            start_idx = random.randint(0, max(len(self.seqclips) - self.max_images, 0))
            self.seqclips = self.seqclips[start_idx : start_idx + self.max_images]

    def __len__(self):
        return len(self.seqclips)

    def __getitem__(self, index):
        transform_rgb = T.Compose([T.Resize((1024, 2048)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        transform_density = T.Compose([T.ToTensor()])
        cur_density_path = self.seqclips[index][0].replace("images", "ground_truth").replace("img", "GT_img").replace("jpg", "h5")
        imgs = torch.stack([transform_rgb(Image.open(img_path).convert("RGB")) for img_path in self.seqclips[index]], dim=1) # [3, 2, 1024, 2048]
        c, t, h, w = imgs.shape
        cur_density = h5py.File(cur_density_path, "r")["density"][:] # [2160, 4096]
        ori_shape = cur_density.shape
        dens = torch.stack([transform_density((cv2.resize(h5py.File(img_path.replace("images", "ground_truth").replace("img", "GT_img").replace("jpg", "h5"),
                                         "r")["density"][:], (w, h), interpolation=cv2.INTER_CUBIC) * (ori_shape[0] * ori_shape[1]) / (w * h)).astype("float32", copy=False))
                                               for img_path in self.seqclips[index]], dim=1) # [1, 2, 1024, 2048]
        result = {}
        if self.image_size is not None:
            while True:
                x1, y1 = random_crop(imgs, self.image_size)
                temp_dens = dens[:, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size]
                if (torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0.5 or (random.random() < 0.2 and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0)
                    or (random.random() < 0.1 and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) == 0) or self.split != "train"):
                    dens = temp_dens
                    imgs = imgs[:, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size]
                    break
        else:
            imgs = imgs
            dens = dens
        if self.random_flip and random.random() < 0.4:
            imgs = torch.flip(imgs, dims=[3])
            dens = torch.flip(dens, dims=[3])
        if self.return_unnorm:
            result["unnorm_density"] = cur_density
        if self.dens_norm:
            dens = torch.stack([T.Normalize(self.MEAN, self.STD)(dens[:, t]) for t in range(dens.shape[1])], dim=1)
        result["rgb"] = imgs # [3, 2, 512, 512]
        result["density"] = dens # [1, 2, 512, 512]
        result["name"] = self.seqclips[index][0].split("/")[-1]
        return result

def random_crop(imgs, crop_size):
    h, w = imgs.shape[-2:]
    th, tw = crop_size, crop_size
    if w == tw and h == th:
        return 0, 0
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return x1, y1

def buildDensityDataset(data_root, split="train", image_size=256, max_images=None, dens_norm=True, random_flip=True, return_unnorm=False, dataset_name="DroneBird", clip_size=3,
                        stride=2, MEAN=[0.0], STD=[1.0]):
    if dataset_name == "DroneBird":
        return DensityDroneBird(data_root, split=split, image_size=image_size, max_images=max_images, dens_norm=dens_norm, random_flip=random_flip, return_unnorm=return_unnorm,
                                clip_size=clip_size, stride=stride, MEAN=MEAN, STD=STD)
    else:
        print('This dataset does not exist')
        raise NotImplementedError