import cv2
import h5py
import numpy as np
from PIL import Image

def load_data(img_path):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint']) # [512, 768]
            img = np.asarray(gt_file['image']) # [600, 800, 3]
            img = Image.fromarray(img, mode='RGB')
            break
        except OSError:
            cv2.waitKey(1000)
    img = img.copy()
    k = k.copy()
    return img, k

def load_data_test(img_path, args, train=True):
    img = Image.open(img_path).convert('RGB')
    return img