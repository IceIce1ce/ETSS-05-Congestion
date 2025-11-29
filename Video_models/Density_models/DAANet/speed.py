import warnings
warnings.filterwarnings("ignore")
from model.video_crowd_flux import DAANet
import argparse
import time
from thop import profile
import numpy as np
import torch

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def test(cfg):
    with torch.no_grad():
        net = DAANet(cfg).cuda()
        state_dict = torch.load(args.ckpt_dir, map_location='cuda')
        try:
            net.load_state_dict(state_dict, strict=True)
        except:
            net.load_state_dict(state_dict["net"], strict=True)
        print('Load ckpt from:', args.ckpt_dir)
        img = torch.rand(2, 3, 768, 1024).cuda()
        macs, params = profile(net, inputs=(img,), verbose=False)
        print("Number of parameter: %.2fM" % (params / 1e6))
        t = 0
        k = 100
        for i in range(k):
            start = time.time()
            _, _, _, _, _, _, _, _, _, _ = net(img)
            end = time.time()
            t += end - start
        e = int((k + 1)) / t
        print('Latency: {:.2f}, FPS: {:.2f}'.format(t, e))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument('--test_intervals', type=int, default=11)
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--mean_std', type=tuple, default=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    # testing config
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--train_size', type=int, nargs='+', default=[768, 1024])
    # model config
    parser.add_argument('--feature_scale', type=float, default=1 / 4.)
    parser.add_argument('--den_factor', type=float, default=200.)
    parser.add_argument('--roi_radius', type=float, default=4.)
    parser.add_argument('--gaussian_sigma', type=float, default=4)
    parser.add_argument('--conf_block_size', type=int, default=16)
    parser.add_argument('--backbone', type=str, default='vgg')
    parser.add_argument('--ckpt_dir', type=str, default='')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    args.val_intervals = args.test_intervals
    args.mode = 'test'
    setup_seed(args.seed)
    test(args)