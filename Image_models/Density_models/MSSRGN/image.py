import h5py
import numpy as np
from PIL import Image
import cv2

def load_data(img_path):
    while True:
        try:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
            img = Image.open(img_path).convert('RGB')
            gt_file = h5py.File(gt_path)
            target = np.asarray(gt_file['density_map']) # [341, 512]
            k = np.asarray(gt_file['kpoint']) # [341, 512]
            sigma_map = np.asarray(gt_file['sigma_map']) # [341, 512]
            img = img.copy()
            target = target.copy()
            sigma_map = sigma_map.copy()
            k = k.copy()
            break
        except OSError:
            cv2.waitKey(5)
    return img, target, k, sigma_map