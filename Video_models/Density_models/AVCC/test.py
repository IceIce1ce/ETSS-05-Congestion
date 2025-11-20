import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.autograd import Variable
from torchvision import transforms
import yaml
import dataset
import numpy as np
from model import CANNet2s, XACANNet2s
from variables import MEAN, STD

def main(args):
    checkpoint = torch.load(args.ckpt_dir, map_location='cpu')
    assert 'config' in checkpoint or args.config is not None, "If the configuration is not into the checkpoint, it should be passed with the --config argument"
    if 'config' in checkpoint:
        configs = checkpoint['config']
        print('Load config from the checkpoint')
    else:
        with open(args.config, 'r') as ymlfile:
            configs = yaml.safe_load(ymlfile)
    if args.test_config is None:
        test_json_path = configs['test_json']
        WIDTH = configs['width']
        HEIGHT = configs['height']
        mask_outputs = configs['use_mask'] if 'use_mask' in configs else False
    else:
        with open(args.test_config, 'r') as ymlfile:
            test_configs = yaml.safe_load(ymlfile)
        test_json_path = test_configs['test_json']
        WIDTH = test_configs['width']
        HEIGHT = test_configs['height']
        mask_outputs = test_configs['use_mask'] if 'use_mask' in test_configs else False
    with open(test_json_path, 'r') as outfile:
        img_paths = json.load(outfile)
    # model
    model_fn = eval(configs['model'])
    model = model_fn(load_weights=False)
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    pred = []
    gt = []
    errs = []
    game = 0
    # test loader
    test_dataset = dataset.listDataset(img_paths, shuffle=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]), train=False, shape=(WIDTH, HEIGHT))
    if not mask_outputs:
        roi = None
    for i in range(len(img_paths)):
        prev_img, img, post_img, prev_target, target, post_target, _ = test_dataset[i] # [3, 360, 640], [3, 360, 640], [3, 360, 640], [45, 80], [45, 80], [45, 80]
        prev_img = prev_img.cuda()
        img = img.cuda()
        img = img.unsqueeze(0) # [1, 3, 360, 640]
        prev_img = prev_img.unsqueeze(0) # [1, 3, 360, 640]
        with torch.no_grad():
            prev_flow = model(prev_img, img) # [1, 10, 45, 80]
            prev_flow_inverse = model(img, prev_img) # [1, 10, 45, 80]
        mask_boundry = torch.zeros(prev_flow.shape[2:]) # [45, 80]
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0
        mask_boundry = Variable(mask_boundry.cuda())
        reconstruction_from_prev = F.pad(prev_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(prev_flow[0, 1, 1:, :], (0, 0, 0, 1)) + \
                                   F.pad(prev_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(prev_flow[0, 3, :, 1:], (0, 1, 0, 0)) + prev_flow[0, 4, :, :] + \
                                   F.pad(prev_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + \
                                   F.pad(prev_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + prev_flow[0, 9, :, :] * mask_boundry # [45, 80]
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :, :] * mask_boundry # [45, 80]
        overall = (reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0 # [45, 80]
        overall = overall.cpu().numpy()
        if mask_outputs:
            overall *= roi
            target *= roi
        pred_sum = overall.sum()
        pred.append(pred_sum)
        gt.append(np.sum(target))
        print('Pred: {:.2f}, GT: {:.2f}'.format(pred_sum, np.sum(target)))
        errs.append(abs(np.sum(target) - pred_sum))
        for k in range(target.shape[0]):
            for j in range(target.shape[1]):
                game += abs(overall[k][j] - target[k][j])
        print('MAE: {:.2f}, RMSE: {:.2f}, GAME: {:.2f}'.format(mean_absolute_error(pred, gt), np.sqrt(mean_squared_error(pred, gt)), game / (i + 1)))
    mae = mean_absolute_error(pred, gt)
    rmse = np.sqrt(mean_squared_error(pred, gt))
    game = game / len(pred)
    print('MAE: {:.2f}, RMSE: {:.2f}, GAME: {:.2f}'.format(mae, rmse, game))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='FDST')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default='saved_fdst/best_1.pth.tar')
    parser.add_argument('--test_config', type=str, default=None)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)