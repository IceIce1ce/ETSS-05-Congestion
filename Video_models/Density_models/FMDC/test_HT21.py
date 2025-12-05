import warnings
warnings.filterwarnings("ignore")
import datasets
from model.video_crowd_count import video_crowd_count
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import numpy as np
import torch
from config import cfg
from importlib import import_module

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def test(cfg_data, args):
    # model
    net = video_crowd_count(cfg, cfg_data)
    # test loader
    test_loader, restore_transform = datasets.loading_testset(args.type_dataset, test_interval=args.test_intervals, mode='test')
    state_dict = torch.load(args.ckpt_dir)
    try:
        net.load_state_dict(state_dict["net"], strict=True)
    except:
        net.load_state_dict(state_dict, strict=True)
    print('Load ckpt from:', args.ckpt_dir)
    net.eval()
    if args.type_dataset == 'HT21':
        gt_flow_cnt = [133, 737, 734, 1040, 321]
    elif args.type_dataset == 'CARLA':
        gt_flow_cnt = [232, 204, 278, 82, 349]
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    scenes_pred_dict = []
    if args.skip_flag:
        intervals = 1
    else:
        intervals = args.test_intervals
    for scene_id, sub_valset in enumerate(test_loader, 0):
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + args.test_intervals
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        for vi, data in enumerate(gen_tqdm, 0):
            img, img_rgb, _ = data
            img, img_rgb = img[0], img_rgb[0]
            img = torch.stack(img, 0).cuda() # [2, 3, 1080, 1920]
            img_rgb = torch.stack(img_rgb, 0).cuda() # [2, 3, 1080, 1920]
            with torch.no_grad():
                b, c, h, w = img.shape
                if h % 64 != 0:
                    pad_h = 64 - h % 64
                else:
                    pad_h = 0
                if w % 64 != 0:
                    pad_w = 64 - w % 64
                else:
                    pad_w = 0
                pad_dims = (0, pad_w, 0, pad_h)
                img = F.pad(img, pad_dims, "constant") # [2, 3, 1088, 1920]
                img_rgb = F.pad(img_rgb, pad_dims, "constant") # [2, 3, 1088, 1920]
                if vi % args.test_intervals == 0 or vi == len(sub_valset) - 1:
                    frame_signal = 'match'
                else:
                    frame_signal = 'skip'
                if frame_signal == 'match' or not args.skip_flag:
                    pred_map, _, pre_outflow, pre_inflow = net.test_or_validate(img, None) # [2, 1, 1088, 1920]
                    pred_cnt = pred_map[0].sum().item()
                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                    pred_dict['inflow'].append(pre_inflow)
                    pred_dict['outflow'].append(pre_outflow)
                if frame_signal == 'match':
                    pre_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, intervals)
                    print('Den pred: %.2f, Pred crowd flow: %.2f, Pred inflow: %.2f' %  (pred_cnt, pre_crowdflow_cnt, pre_inflow))
        scenes_pred_dict.append(pred_dict)
    MAE, MSE, WRAE, crowdflow_cnt = compute_metrics_all_scenes(scenes_pred_dict, gt_flow_cnt, intervals)
    print('MAE: {:.2f}, MSE: {:.2f}, WRAE: {:.2f}, Crowd flow count: {}'.format(MAE.data, MSE.data, WRAE.data, crowdflow_cnt))

def compute_metrics_single_scene(pre_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt =torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2)
    pre_crowdflow_cnt  = pre_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'], pre_dict['outflow']), 0):
        inflow_cnt[idx, 0] = data[0]
        outflow_cnt[idx, 0] = data[1]
        if idx %intervals == 0 or idx == len(pre_dict['inflow']) - 1:
            pre_crowdflow_cnt += data[0]
    return pre_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE': torch.zeros(scene_cnt, 2), 'WRAE': torch.zeros(scene_cnt, 2)}
    for i, (pre_dict, gt_dict) in enumerate(zip(scenes_pred_dict, scene_gt_dict), 0):
        time = pre_dict['time']
        gt_crowdflow_cnt = gt_dict
        pre_crowdflow_cnt, inflow_cnt, outflow_cnt = compute_metrics_single_scene(pre_dict, intervals)
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i, :] = torch.tensor([mae / (gt_crowdflow_cnt + 1e-10), time])
    MAE =  torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1])**2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:, 0] * (metrics['WRAE'][:, 1] / (metrics['WRAE'][:, 1].sum() + 1e-10))) * 100
    return MAE, MSE, WRAE, metrics['MAE']

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='HT21')
    parser.add_argument('--test_intervals', type=int, default=60)
    parser.add_argument('--skip_flag', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--ckpt_dir', type=str, default='saved_den_ht21/ep_1_iter_1000_mae_30.234_mse_30.726_seq_MAE_220.805_WRAE_253.799_MIAE_21.129_MOAE_22.114.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    setup_seed(args.seed)
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    mae, mse, wrae = [], [], []
    test(datasetting.cfg_data, args)
