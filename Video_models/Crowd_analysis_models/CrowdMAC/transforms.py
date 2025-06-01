import numbers
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def create_train_augmentation_list(cfg):
    augmentation_list = []
    augmentation_list += [Stack(roll=False)]
    augmentation_list += [ToTorchFormatTensor(div=False)]
    if cfg.augmentation.train.RandomHorizontalFlip.apply:
        augmentation_list += [RandomHorizontalFlip(p=cfg.augmentation.train.RandomHorizontalFlip.p)]
    if cfg.augmentation.train.RandomVerticalFlip.apply:
        augmentation_list += [RandomVerticalFlip(p=cfg.augmentation.train.RandomVerticalFlip.p)]
    return augmentation_list

def create_test_augmentation_list(cfg):
    augmentation_list = []
    augmentation_list += [Stack(roll=False)]
    augmentation_list += [ToTorchFormatTensor(div=False)]
    return augmentation_list

class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        if img_group[0].mode == "L":
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)
        elif img_group[0].mode == "RGB":
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate(img_group, axis=2), label)
        elif img_group[0].mode == "F":
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)

class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return (img.float().div(255.0) if self.div else img.float(), label)

class IdentityTransform(object):
    def __call__(self, data):
        return data

class GroupRandomRotation(object):
    def __init__(self, degree=180):
        self.worker = transforms.RandomRotation(degrees=degree)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)

class RandomRotation(object):
    def __init__(self, degrees=180):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number," "must be positive")
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence," "it must be of len 2.")
        self.degrees = degrees

    def __call__(self, img_tuple):
        import skimage
        img_group, label = img_tuple
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(img_group[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in img_group]
        elif isinstance(img_group[0], Image.Image):
            rotated = [img.rotate(angle) for img in img_group]
        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(img_group[0])))
        return rotated, label

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor_tuple):
        img_group, label = tensor_tuple
        if np.random.uniform() < self.p:
            img_group = img_group.flip((-1))
        return (img_group, label)

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor_tuple):
        img_group, label = tensor_tuple
        if np.random.uniform() < self.p:
            img_group = img_group.flip((-2))
        return (img_group, label)