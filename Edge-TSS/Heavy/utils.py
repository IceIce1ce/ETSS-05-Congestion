import logging
import os
import random
import shutil
import cv2
import numpy as np
import torch
import torch.distributed as dist

logger_initialized = {}

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    return logger

def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    density_map[density_map < 0] = 0
    gt_data = 255 * gt_data / np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)
    gt_data = cv2.applyColorMap(gt_data, 2)
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)
    result_img = np.hstack((gt_data, density_map))
    cv2.imwrite(os.path.join('.', output_dir, fname).replace('.jpg', '.jpg'), result_img)

def save_checkpoint(state, visi, is_best, save_path, filename='checkpoint.pth'):
    torch.save(state, './' + str(save_path) + '/' + filename)
    if is_best:
        shutil.copyfile('./' + str(save_path) + '/' + filename, './' + str(save_path) + '/' + 'model_best.pth')
    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, str(save_path), fname[0])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='CLTR', log_file=log_file, log_level=log_level)
    return logger