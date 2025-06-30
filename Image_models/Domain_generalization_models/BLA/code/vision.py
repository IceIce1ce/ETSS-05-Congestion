import cv2
import os
import numpy as np
import torch

def save_keypoints_and_img(img, den_map, opt, save_filename, img_name): # [3, 704, 1024], [704, 1024]
    if not os.path.exists(os.path.split(save_filename)[0]):
        os.makedirs(os.path.split(save_filename)[0])
    file_dir = os.path.split(save_filename)[0]
    mean = opt.mean_std[0]
    std = opt.mean_std[1]
    new_img = torch.zeros(img.size())
    for i in range(3):
        new_img[i] = img[i] * std[i] + mean[i]
    new_img *= 255 # [3, 704, 1024]
    new_img = new_img.numpy()
    new_img = new_img.astype(np.uint8)
    new_img = new_img.transpose([1, 2, 0])
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR) # [704, 1024, 3]
    if not os.path.exists(os.path.join(file_dir,img_name)):
        try:
            cv2.imwrite(os.path.join(file_dir, img_name), new_img)
        except:
            pass
    den_map = (den_map - torch.min(den_map)) / (torch.max(den_map) - torch.min(den_map)) # [704, 1024]
    den_map = den_map.unsqueeze(2) # [704, 1024, 1]
    den_map = den_map.cpu().numpy() * 255 # [704, 1024, 1]
    den_map = den_map.astype(np.uint8) # [704, 1024, 1]
    vision_map = cv2.applyColorMap(den_map, cv2.COLORMAP_JET) # [704, 1024, 3]
    try:
        cv2.imwrite(save_filename, vision_map)
    except:
        pass