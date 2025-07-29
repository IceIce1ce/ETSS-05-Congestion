import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from models.UGSDA import UGSDA
import argparse
import warnings
warnings.filterwarnings("ignore")

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count

def test(file_list, args):
    # model
    net = UGSDA()
    net.cuda()
    net.load_state_dict(torch.load(args.ckpt_dir, map_location='cpu'), strict=False)
    net.eval()
    # metrics
    maes = AverageMeter()
    mses = AverageMeter()
    for filename in file_list:
        imgname = os.path.join(args.input_dir, 'img', filename)
        filename_no_ext = filename.split('.')[0]
        denname = args.input_dir + '/den/' + filename_no_ext + '.csv'
        den = pd.read_csv(denname, sep=',', header=None).values
        den = den.astype(np.float32, copy=False)
        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img) # [3, 1280, 1920]
        gt = np.sum(den) # [1]
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda() # [1, 3, 1280, 1920]
            pred_map = net.test_forward(img) # [1, 1, 1280, 1920]
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :] # [1280, 1920]
        pred = np.sum(pred_map) / args.log_para # [1]
        maes.update(abs(gt - pred))
        mses.update(((gt - pred) * (gt - pred)))
    mae = maes.avg
    mse = np.sqrt(mses.avg)
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='qnrf')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/SHHA_parameters.pth')
    parser.add_argument('--input_dir', type=str, default='datasets/UCF-QNRF/test_data')
    parser.add_argument('--log_para', type=float, default=100.0)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)])
    file_list = [filename for root, dirs, filename in os.walk(args.input_dir + '/img/')]
    test(file_list[0], args)