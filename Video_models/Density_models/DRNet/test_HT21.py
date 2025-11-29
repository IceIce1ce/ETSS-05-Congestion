import warnings
warnings.filterwarnings('ignore')
import datasets
from misc.utils import make_matching_plot_fast
import cv2
from model.VIC import Video_Individual_Counter
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import argparse
import matplotlib.cm as cm
import numpy as np
import torch
from config import cfg
from importlib import import_module

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def test(cfg_data):
    # model
    net = Video_Individual_Counter(cfg, cfg_data)
    # test loader
    test_loader, restore_transform = datasets.loading_testset(args.type_dataset, test_interval=args.test_intervals, mode='test')
    state_dict = torch.load(args.ckpt_dir)
    net.load_state_dict(state_dict, strict=True)
    print('Load ckpt from:', args.ckpt_dir)
    net.eval()
    gt_flow_cnt = [133, 737, 734, 1040, 321]
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
            img, _ = data
            img = img[0]
            img = torch.stack(img, 0)
            with torch.no_grad():
                b, c, h, w = img.shape
                if h % 16 != 0:
                    pad_h = 16 - h % 16
                else:
                    pad_h = 0
                if w % 16 != 0:
                    pad_w = 16 - w % 16
                else:
                    pad_w = 0
                pad_dims = (0, pad_w, 0, pad_h)
                img = F.pad(img, pad_dims, "constant")
                if vi % args.test_intervals == 0 or vi == len(sub_valset) - 1:
                    frame_signal = 'match'
                else:
                    frame_signal = 'skip'
                if frame_signal == 'match' or not args.skip_flag:
                    pred_map, matched_results = net.test_forward(img)
                    pred_cnt = pred_map[0].sum().item()
                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                    pred_dict['inflow'].append(matched_results['pre_inflow'])
                    pred_dict['outflow'].append(matched_results['pre_outflow'])
                if frame_signal == 'match':
                    pre_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, intervals)
                    print('Den pred: {:.2f}, Pred crowd flow: {:.2f}, Pred inflow: {:.2f}'.format(pred_cnt, pre_crowdflow_cnt, matched_results['pre_inflow']))
                    kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
                    kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()
                    matches = matched_results['matches0'].cpu().numpy()
                    confidence = matched_results['matching_scores0'].cpu().numpy()
                    if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                        save_visImg(kpts0, kpts1, matches, confidence, vi, img[0].clone(), img[1].clone(), args.test_intervals, args.output_dir, None, None, scene_id, restore_transform)
        scenes_pred_dict.append(pred_dict)
    MAE, MSE, WRAE, crowdflow_cnt = compute_metrics_all_scenes(scenes_pred_dict, gt_flow_cnt, intervals)
    print('MAE: {:.2f}, MSE: {:.2f}, WRAE: {:.2f}, Crowd flow count: {}'.format(MAE.data, MSE.data, WRAE.data, crowdflow_cnt))
    return MAE, MSE, WRAE

def compute_metrics_single_scene(pre_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt = torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2)
    pre_crowdflow_cnt = pre_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'], pre_dict['outflow']), 0):
        inflow_cnt[idx, 0] = data[0]
        outflow_cnt[idx, 0] = data[1]
        if idx % intervals == 0 or  idx == len(pre_dict['inflow']) - 1:
            pre_crowdflow_cnt += data[0]
    return pre_crowdflow_cnt, inflow_cnt, outflow_cnt

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

def save_visImg(kpts0, kpts1, matches, confidence, vi, last_frame, cur_frame, intervals, save_path, id0=None, id1=None, scene_id='', restore_transform=None):
    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1, 2)
    mkpts1 = kpts1[matches[valid]].reshape(-1, 2)
    color = cm.jet(confidence[valid])
    out, out_by_point = make_matching_plot_fast(last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, path=None, show_keypoints=True, restore_transform=restore_transform, id0=id0, id1=id1)
    if save_path is not None:
        Path(save_path).mkdir(exist_ok=True)
        stem = '{}_{}_{}_matches'.format(scene_id, vi, vi + intervals)
        out_file = str(Path(save_path, stem + '.png'))
        print('Writing image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        out_file = str(Path(save_path, stem + '_vis.png'))
        cv2.imwrite(out_file, out_by_point)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='HT21')
    parser.add_argument('--output_dir', type=str, default='saved_ht21')
    parser.add_argument('--test_intervals', type=int, default=75)
    parser.add_argument('--skip_flag', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--ckpt_dir', type=str, default='saved_ht21/HT21.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    setup_seed(args.seed)
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    mae, mse, wrae = [], [], []
    MAE, MSE, WRAE = test(datasetting.cfg_data)
    mae.append(MAE.item())
    mse.append(MSE.item())
    wrae.append(WRAE.item())
    print('Average MAE: {:.2f}, MSE: {:.2f}, WRAE: {:.2f}'.format(np.mean(mae), np.mean(mse), np.mean(wrae)))