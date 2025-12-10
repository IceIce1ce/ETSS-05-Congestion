from PIL import Image
import numpy as np
import h5py
import cv2

def load_data_fidt(img_path):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map')
    img = Image.open(img_path).convert('RGB')
    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            fidt_map = np.asarray(gt_file['fidt_map'])
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)
    img = img.copy() # [768, 1024, 3]
    fidt_map = fidt_map.copy() # [768, 1024]
    k = k.copy() # [768, 1024]
    return img, fidt_map, k