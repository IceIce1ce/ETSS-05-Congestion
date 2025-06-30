import networks
import torch
import argparse
from main import get_data, get_test_loader
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/UCF-QNRF', type=str)
    parser.add_argument('--model_dir', default='SHA.pth.tar', type=str)
    args = parser.parse_args()

    print('Testing dataset:', args.input_dir.split('/')[-1])
    # model
    net = networks.create('memMeta')
    checkpoint = torch.load(args.model_dir, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint, strict=False)
    net.cuda()
    net.eval()
    val_loss = []
    mae = 0.0
    mse = 0.0
    # test loader
    test_set = get_data(args.input_dir, source=False)
    test_loader = get_test_loader(test_set, 1, 4)
    for vi, data in enumerate(test_loader, 0):
        img, gt_map = data # [1, 3, 1280, 1920], [1, 1280, 1920]
        with torch.no_grad():
            img = img.cuda() # [1, 3, 1280, 1920]
            gt_map = gt_map.cuda() # [1, 1280, 1920]
            pred_map = net(img) # [1, 1, 1280, 1920]
            pred_map = pred_map.data.cpu().numpy() # [1, 1, 1280, 1920]
            gt_map = gt_map.data.cpu().numpy() # [1, 1280, 1920]
            gt_count = np.sum(gt_map) / 1000.
            pred_cnt = np.sum(pred_map) / 1000.
            mae += abs(gt_count - pred_cnt)
            mse += ((gt_count - pred_cnt) * (gt_count - pred_cnt))
    mae = mae / len(test_loader)
    mse = np.sqrt(mse / len(test_loader))
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))