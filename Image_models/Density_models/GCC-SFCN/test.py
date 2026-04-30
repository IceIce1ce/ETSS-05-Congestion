from matplotlib import pyplot as plt
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import pandas as pd
from models.CC import CrowdCounter
from config import cfg
import numpy as np
import argparse
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

def test(file_list, model_path, args):
    net = CrowdCounter()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    maes = []
    mses = []
    for filename in file_list:
        imgname = args.input_dir + '/img/' + filename
        filename_no_ext = filename.split('.')[0]
        denname = args.input_dir + '/den/' + filename_no_ext + '.csv'
        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False) # [685, 1024]
        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        wd_1, ht_1 = img.size
        if wd_1 < cfg.DATA.STD_SIZE[1]:
            dif = cfg.DATA.STD_SIZE[1] - wd_1
            img = ImageOps.expand(img, border=(0,0,dif,0), fill=0)
            pad = np.zeros([ht_1,dif])
            den = np.array(den)
            den = np.hstack((den,pad))
        if ht_1 < cfg.DATA.STD_SIZE[0]:
            dif = cfg.DATA.STD_SIZE[0] - ht_1
            img = ImageOps.expand(img, border=(0,0,0,dif), fill=0)
            pad = np.zeros([dif,wd_1])
            den = np.array(den)
            den = np.vstack((den,pad))
        img = img_transform(img) # [3, 768, 1024]
        gt = np.sum(den)
        img = Variable(img[None, :, :, :], volatile=True).cuda() # [1, 3, 768, 1024]
        pred_map = net.test_forward(img) # [1, 1, 768, 1024]
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :] # [768, 1024]
        pred = np.sum(pred_map) / 100.0
        maes.append(abs(pred - gt))
        mses.append((pred - gt) * (pred - gt))
        pred_map = pred_map / np.max(pred_map + 1e-20) # [768, 1024]
        pred_map = pred_map[0:ht_1, 0:wd_1] # [685, 1024]
        den = den / np.max(den + 1e-20) # [768, 1024]
        den = den[0:ht_1, 0:wd_1] # [685, 1024]
        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False) 
        den_frame.spines['bottom'].set_visible(False) 
        den_frame.spines['left'].set_visible(False) 
        den_frame.spines['right'].set_visible(False) 
        plt.savefig(args.output_dir + '/' + filename_no_ext + '_gt_' + str(int(gt)) + '.png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False) 
        pred_frame.spines['bottom'].set_visible(False) 
        pred_frame.spines['left'].set_visible(False) 
        pred_frame.spines['right'].set_visible(False) 
        plt.savefig(args.output_dir + '/' + filename_no_ext + '_pred_' + str(float(pred)) + '.png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        diff = den - pred_map # [685, 1024]
        diff_frame = plt.gca()
        plt.imshow(diff, 'jet')
        plt.colorbar()
        diff_frame.axes.get_yaxis().set_visible(False)
        diff_frame.axes.get_xaxis().set_visible(False)
        diff_frame.spines['top'].set_visible(False) 
        diff_frame.spines['bottom'].set_visible(False) 
        diff_frame.spines['left'].set_visible(False) 
        diff_frame.spines['right'].set_visible(False) 
        plt.savefig(args.output_dir + '/' + filename_no_ext + '_diff.png', bbox_inches='tight',pad_inches=0, dpi=100)
        plt.close()
        print('Name: {}, Pred: {:.4f}, GT: {:.4f}'.format(filename, pred, gt))
    mae = np.average(np.array(maes))
    mse = np.sqrt(np.average(np.array(mses)))
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='datasets/qnrf/test')
    parser.add_argument('--output_dir', default='saved_vis', type=str)
    parser.add_argument('--ckpt_dir', type=str, default='best.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.input_dir.split('/')[1])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    mean_std = cfg.DATA.MEAN_STD
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)])
    file_list = [filename for root, dirs, filename in os.walk(args.input_dir + '/img/')]
    test(file_list[0], args.ckpt_dir, args)