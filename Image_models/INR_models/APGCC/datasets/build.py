import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
from typing import List
from .dataset import ImageDataset

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def loading_data(cfg):
    transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_set = ImageDataset(cfg.DATASETS.DATA_ROOT, train=True, transform=transform, aug_dict=cfg.DATALOADER)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, cfg.SOLVER.BATCH_SIZE, drop_last=True)
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train, collate_fn=collate_fn_crowd, num_workers=cfg.DATALOADER.NUM_WORKERS)
    val_set = ImageDataset(cfg.DATASETS.DATA_ROOT, train=False, transform=transform, aug_dict=cfg.DATALOADER)
    sampler_val = torch.utils.data.SequentialSampler(val_set)
    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val, drop_last=False, collate_fn=collate_fn_crowd, num_workers=cfg.DATALOADER.NUM_WORKERS)
    return data_loader_train, data_loader_val

def collate_fn_crowd(batch):
    batch_new = []
    for b in batch:
        imgs, points = b # [4, 3, 128, 128], [51, 2] * len(4)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i, :, :, :], points[i]))
    batch = batch_new
    batch = list(zip(*batch)) 
    batch[0] = _nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def _max_by_axis_pad(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    block = 128
    for i in range(2):
        maxes[i + 1] = ((maxes[i + 1] - 1) // block + 1) * block
    return maxes

def _nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor