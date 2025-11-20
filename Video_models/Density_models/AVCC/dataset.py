import random
from torch.utils.data import Dataset
import torchvision
from image import load_data
from variables import PATCH_SIZE_PF
import numpy as np
import cv2

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, mask=None, debug=False, output_original_target=False):
        if shuffle:
            random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.debug = debug
        self.mask = mask if mask is not None else np.ones(shape).T
        self.output_original_target = output_original_target

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        prev_img, img, post_img, prev_target, target, post_target = load_data(img_path, self.shape)
        if self.transform is not None:
            if isinstance(self.transform, torchvision.transforms.Compose):
                prev_img = self.transform(prev_img)
                img = self.transform(img)
                post_img = self.transform(post_img)
                mask = self.mask
            else:
                prev_img, img, post_img = np.array(prev_img), np.array(img), np.array(post_img)
                transformed = self.transform(image=prev_img, image1=img, image2=post_img, density=prev_target, density1=target, density2=post_target, mask=self.mask)
                prev_img = transformed['image']
                img = transformed['image1']
                post_img = transformed['image2']
                prev_target = transformed['density']
                target = transformed['density1']
                post_target = transformed['density2']
                mask = transformed['mask']
        old_pixel_sum = target.sum()
        target_resized = cv2.resize(target, (int(target.shape[1] / PATCH_SIZE_PF), int(target.shape[0] / PATCH_SIZE_PF)), interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)
        new_pixel_sum = target_resized.sum()
        target_resized = target_resized * old_pixel_sum / new_pixel_sum if old_pixel_sum != 0 else target
        old_pixel_sum = prev_target.sum()
        prev_target = cv2.resize(prev_target, (int(prev_target.shape[1] / PATCH_SIZE_PF), int(prev_target.shape[0] / PATCH_SIZE_PF)), interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)
        new_pixel_sum = prev_target.sum()
        prev_target = prev_target * old_pixel_sum / new_pixel_sum if old_pixel_sum != 0 else prev_target
        old_pixel_sum = post_target.sum()
        post_target = cv2.resize(post_target, (int(post_target.shape[1] / PATCH_SIZE_PF), int(post_target.shape[0] / PATCH_SIZE_PF)), interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)
        new_pixel_sum = post_target.sum()
        post_target = post_target * old_pixel_sum / new_pixel_sum if old_pixel_sum != 0 else post_target
        resized_mask = cv2.resize(mask, (int(self.mask.shape[1] / PATCH_SIZE_PF), int(self.mask.shape[0] / PATCH_SIZE_PF)), interpolation=cv2.INTER_NEAREST)
        if not isinstance(self.transform, torchvision.transforms.Compose) and not self.debug:
            tot = torchvision.transforms.ToTensor()
            img = tot(img)
            prev_img = tot(prev_img)
            post_img = tot(post_img)
        if self.output_original_target:
            return prev_img, img, post_img, prev_target, target_resized, target, post_target, resized_mask
        return prev_img, img, post_img, prev_target, target_resized, post_target, resized_mask