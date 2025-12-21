from PIL import Image, ImageOps
import torch

class RandomHorizontallyFlip(object):
    def __init__(self, task=None):
        self.task = task

    def __call__(self, img, gt, flip_flag=0, bbx=None):
        if flip_flag :
            w, h = img.size
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt['points'][:, 0] = w - gt['points'][:, 0]
        return img, gt

class ScaleByRateWithMin(object):
    def __init__(self, min_w, min_h, task=None):
        self.min_w = min_w
        self.min_h = min_h
        self.task = task

    def __call__(self, img, gt):
        w, h = img.size
        new_w = self.min_w
        new_h = self.min_h
        img = img.resize((new_w, new_h), Image.LANCZOS)
        rate = new_w / w
        gt['points'] =  gt['points']  * rate
        return img, gt

def check_image(img, target, crop_size, max_size):
    w, h = img.size
    long_side = max(w, h)
    short_side = min(w, h)
    max_long_side, max_short_side = max_size
    scale_long = max_long_side / long_side
    scale_short = max_short_side / short_side
    if scale_long < 1 or scale_short < 1:
        scale = min(scale_long, scale_short)
        new_width = int(w * scale)
        new_height = int(h * scale)
        target['points'] = target['points'] * scale
        img = img.resize((new_width, new_height), Image.LANCZOS)
    c_h, c_w = crop_size
    w, h = img.size
    if w < c_w or h < c_h:
        delta_w = max(c_w - w, 0)
        delta_h = max(c_h - h, 0)
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        target['points'] = target['points'] + torch.tensor([delta_w // 2, delta_h // 2], dtype=torch.float32)
    return  img, target

class RandomCrop(object):
    def __init__(self):
        pass

    def __call__(self, img, gt, crop_left, crop_size ):
        th, tw = crop_size[0], crop_size[1]
        x1, y1 = crop_left
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        index = (gt['points'][:, 0] > x1 + 1) & (gt['points'][:, 0] < x1 + tw - 1) & (gt['points'][:, 1] > y1 + 1) & (gt['points'][:, 1] < y1 + th - 1)
        gt['points'] = gt['points'][index].view(-1, 2).contiguous()
        gt['points'] -= torch.tensor([x1, y1], dtype = torch.float32)
        if 'person_id' in gt:
            gt['person_id'] =  gt['person_id'][index]
        elif 'inflow' in gt:
            gt['inflow'] = gt['inflow'][index]
            gt['outflow'] = gt['outflow'][index]
        else:
            raise "error!"
        return img, gt

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor