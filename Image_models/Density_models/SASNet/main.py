import numpy as np
import torch
import argparse
from model import SASNet
from datasets.loading_data import loading_data
import warnings
warnings.filterwarnings('ignore')

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

def main(args):
    # test loader
    test_loader = loading_data(args)
    # load trained model
    model = SASNet(args=args).cuda()
    model.load_state_dict(torch.load(args.ckpt_dir))
    print('Load ckpt from: {}'.format(args.ckpt_dir))
    with torch.no_grad():
        model.eval()
        maes = AverageMeter()
        mses = AverageMeter()
        for vi, data in enumerate(test_loader, 0):
            img, gt_map = data # [1, 3, 350, 1024], [1, 350, 1024]
            img = img.cuda()
            gt_map = gt_map.type(torch.FloatTensor).unsqueeze(0).cuda()
            pred_map = model(img) # [1, 1, 350, 1024]
            pred_map = pred_map.data.cpu().numpy()
            gt_map = gt_map.data.cpu().numpy()
            for i_img in range(pred_map.shape[0]):
                pred_cnt = np.sum(pred_map[i_img]) / args.log_para
                gt_count = np.sum(gt_map[i_img])
                maes.update(abs(gt_count - pred_cnt))
                mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/SHHA.pth')
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A_final')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--log_para', type=int, default=1000)
    parser.add_argument('--block_size', type=int, default=32,)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)