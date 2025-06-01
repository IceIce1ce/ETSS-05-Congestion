import torch
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from datasets.SHHA.setting import cfg_data
from datasets.SHHA.SHHA import SHHA
import numpy as np
from HMoDE import HMoDE
import argparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A_final')
    parser.add_argument('--log_para', type=float, default=100.)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default='saved_sha/best_model.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.input_dir.split('/')[1])
    cfg_data.DATA_PATH = args.input_dir
    cfg_data.RESUME_MODEL = args.ckpt_dir
    # test loader
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(*cfg_data.MEAN_STD)])
    gt_transform = standard_transforms.Compose([own_transforms.LabelNormalize(args.log_para)])
    test_set = SHHA(cfg_data.DATA_PATH + '/test_data', main_transform=None, img_transform=img_transform, gt_transform=gt_transform, data_augment=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # model
    net = HMoDE(False)
    net.load_state_dict(torch.load(cfg_data.RESUME_MODEL)['model'])
    net.cuda()
    net.eval()
    val_loss = []
    mae = 0.0
    mse = 0.0
    for vi, data in enumerate(test_loader, 0):
        img, gt_map = data # [1, 3, 687, 1024], [1, 687, 1024]
        with torch.no_grad():
            img = img.cuda()
            gt_map = gt_map.cuda()
            pred_map = net(img)[0][-1] # [1, 1, 687, 1024]
            pred_map = pred_map.data.cpu().numpy()
            gt_map = gt_map.data.cpu().numpy()
            gt_count = np.sum(gt_map) / args.log_para
            pred_cnt = np.sum(pred_map) / args.log_para
            mae += abs(gt_count - pred_cnt)
            mse += ((gt_count - pred_cnt) * (gt_count - pred_cnt))
    mae = mae / test_set.get_num_samples()
    mse = np.sqrt(mse / test_set.get_num_samples())
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))
