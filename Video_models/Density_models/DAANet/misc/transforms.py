from PIL import Image, ImageOps
import torch

class RandomHorizontallyFlip(object):
    def __call__(self, img, gt, flip_flag=0, bbx=None):
        if flip_flag :
            w, h = img.size
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt['points'][:,0] =w -  gt['points'][:,0]
        return img, gt

class ScaleByRateWithMin(object):
    def __init__(self, min_w, min_h):
        self.min_w = min_w
        self.min_h = min_h

    def __call__(self, img, gt=None):
        w, h = img.size
        new_w = self.min_w
        new_h = self.min_h
        img = img.resize((new_w, new_h), Image.LANCZOS)
        rate = new_w / w
        if gt != None:
            gt['points'] =  gt['points'] * rate
            gt['sigma']  =  gt['sigma']  * rate
            return img, gt
        else:
            return img

def check_image(img, target, crop_size):
    w, h = img.size
    c_h, c_w = crop_size
    if w < c_w or h < c_h:
        delta_w = max(c_w - w, 0)
        delta_h = max(c_h - h, 0)
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        target['points'] = target['points'] + torch.tensor([delta_w // 2, delta_h // 2], dtype = torch.float32)
    return  img, target

class RandomCrop(object):
    def __call__(self, img, gt, crop_left, crop_size):
        th, tw = crop_size[0], crop_size[1]
        x1,y1 = crop_left
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        index = (gt['points'][:,0]>x1+1) & (gt['points'][:,0]<x1 + tw-1) & (gt['points'][:,1]>y1+1) & (gt['points'][:,1]<y1 + th-1)
        gt['points'] = gt['points'][index].view(-1,2).contiguous()
        gt['points'] -= torch.tensor([x1, y1], dtype = torch.float32)
        gt['person_id'] =  gt['person_id'][index]
        gt['sigma'] = gt['sigma'][index]
        return img, gt

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor