import os
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from PIL import Image
from models.UGSDA import UGSDA
from matplotlib import pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

def test_vis_single(args):
    # model
    net = UGSDA()
    net.cuda()
    net.load_state_dict(torch.load(args.ckpt_dir, map_location=lambda storage, loc: storage), strict=False)
    net.eval()
    img = Image.open(args.input_dir)
    if img.mode == 'L':
        img = img.convert('RGB')
    img = img_transform(img)
    with torch.no_grad():
        img = Variable(img[None, :, :, :]).cuda()
        pred_map = net.test_forward(img)
    pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]
    pred = np.sum(pred_map) / args.log_para
    pred_map = pred_map / np.max(pred_map + 1e-20)
    print("Count: {:.4f}".format(pred))
    den_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    den_frame.axes.get_yaxis().set_visible(False)
    den_frame.axes.get_xaxis().set_visible(False)
    den_frame.spines['top'].set_visible(False) 
    den_frame.spines['bottom'].set_visible(False) 
    den_frame.spines['left'].set_visible(False) 
    den_frame.spines['right'].set_visible(False)
    plt.savefig(os.path.join(args.output_dir, args.input_dir.split('/')[-1]), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='qnrf')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/SHHA_parameters.pth')
    parser.add_argument('--input_dir', type=str, default='datasets/UCF-QNRF/test_data/img/1.jpg')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--log_para', type=float, default=100.0)
    args = parser.parse_args()

    print('Visualize dataset:', args.type_dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)])
    test_vis_single(args)