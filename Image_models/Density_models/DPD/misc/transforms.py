import numbers
import random
import numpy as np
from PIL import Image, ImageOps
import torch
import os
import torch.nn.functional as F
from torchvision import transforms as tf
from PIL import ImageEnhance

def exact_feature_distribution_matching(content, tf, args):
    tra_root = os.path.join(args.target_dir, 'images')
    tra_lst = os.listdir(tra_root)
    tra_img = np.random.randint(0, len(tra_lst))
    tra_img = Image.open(os.path.join(tra_root, tra_lst[tra_img])).convert('RGB')
    style = tf(tra_img) # [1, 3, 1536, 2048]
    B, C, W, H = content.size(0), content.size(1), content.size(2), content.size(3)
    if not (content.size() == style.size()):
        style = F.interpolate(style, (W, H))
    _, index_content = torch.sort(content.view(B, C, -1))
    value_style, _ = torch.sort(style.view(B, C, -1))
    inverse_index = index_content.argsort(-1)
    transferred_content = content.view(B, C, -1) + value_style.gather(-1, inverse_index) - content.view(B, C, -1).detach()
    return transferred_content.view(B, C, W, H) # [1, 3, 1536, 2048]

def Brightness(image):
    enh_bri = ImageEnhance.Brightness(image)
    brightness = np.random.randint(0, 20) / 10
    image = enh_bri.enhance(brightness)
    return image

def Chromaticity(image):
    enh_col = ImageEnhance.Color(image)
    color = np.random.randint(0, 20) / 10
    image = enh_col.enhance(color)
    return image

def Contrast(image):
    enh_con = ImageEnhance.Contrast(image)
    contrast = np.random.randint(0, 20) / 10
    image = enh_con.enhance(contrast)
    return image

def Sharpness(image):
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = np.random.randint(0, 30) / 10
    image = enh_sha.enhance(sharpness)
    return image

class RandomAugment(object):
    def __init__(self, args):
        self.args = args
        self.Candidates = ['Brightness', 'Chromaticity', 'Contrast', 'Sharpness', 'EFDM', 'Noise']
        self.tf = tf.Compose([tf.ToTensor(), lambda x: x * 255, lambda x: x.unsqueeze(0)])
        self.re_tf = tf.Compose([lambda x: x.squeeze(0), lambda x: x.numpy(), lambda x: np.transpose(x, (1, 2, 0)), lambda x: Image.fromarray(x, mode='RGB')])

    def __call__(self, img):
        num_process = np.random.randint(1, len(self.Candidates) + 1)
        chosen_ones = np.random.choice(self.Candidates, num_process)
        specialize = False
        for one in chosen_ones:
            if one not in ['EFDM', 'Noise']:
                img = eval(f'{one}(img)')
            else:
                specialize = True
        if specialize:
            img = self.tf(img)
            if 'EFDM' in self.Candidates:
                img = exact_feature_distribution_matching(img, self.tf, self.args)
            else:
                noise_ratio = 0.2
                img = noise_ratio + img
            img = self.re_tf(img)
        return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img,mask = t(img, mask)
            return img,mask
        for t in self.transforms:
            img,mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            xmin = w - bbx[:, 3]
            xmax = w - bbx[:, 1]
            bbx[:, 1] = xmin
            bbx[:, 3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img, mask
        return img, mask, bbx

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, dst_size=None ):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
        assert img.size == mask.size
        w, h = img.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return img, mask
        assert w >= tw
        assert h >= th
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class ScaleByRateWithMin(object):
    def __init__(self, rateRange, min_w, min_h):
        self.rateRange = rateRange
        self.min_w = min_w
        self.min_h = min_h

    def __call__(self, img, mask):
        w, h = img.size
        rate = random.uniform(self.rateRange[0], self.rateRange[1])
        new_w = int(w * rate) // 32 * 32
        new_h = int(h * rate) // 32 * 32
        if new_h< self.min_h or new_w<self.min_w:
            if new_w < self.min_w:
                new_w = self.min_w
                rate = new_w / w
                new_h = int(h * rate) // 32 * 32
            if new_h < self.min_h:
                new_h = self.min_h
                rate = new_h / h
                new_w = int( w * rate) // 32 * 32
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return img, mask

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor