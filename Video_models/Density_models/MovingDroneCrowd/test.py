import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm
from copy import deepcopy
import datasets
from misc.utils import AverageMeter, save_test_visual
import torch.nn.functional as F
from model.VIC import Video_Counter
from train import compute_metrics_all_scenes
import os
import numpy as np
import torch
from config import cfg
from importlib import import_module

def setup_seed():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        state_dict[k] = v
    return state_dict

def test(cfg_data):
    # model
    model = Video_Counter(cfg, cfg_data)
    model.cuda()
    with open(os.path.join(cfg_data.DATA_PATH, 'scene_label.txt'), 'r') as f:
        lines = f.readlines()
    scene_label = {}
    for line in lines:
        line = line.rstrip().split(' ')
        scene_label.update({line[0]: [int(i) for i in line[1:]] })
    # test loader
    test_loader, restore_transform = datasets.loading_testset(args.type_dataset, args.test_intervals, args.skip_flag, mode='test')
    state_dict = torch.load(args.ckpt_dir, map_location='cpu')
    model.load_state_dict(module2model(state_dict), strict=True)
    print('Load ckpt from:', args.ckpt_dir)
    model.eval()
    sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
    scenes_pred_dict = {'all': [], 'density0': [],'density1': [],'density2': [], 'density3': []}
    scenes_gt_dict =  {'all': [], 'density0': [],'density1': [],'density2': [], 'density3': []}
    if args.skip_flag:
        intervals = 1
    else:
        intervals = args.test_intervals
    for scene_id, (scene_name, sub_valset) in enumerate(test_loader, 0):
        test_interval = args.test_intervals
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + test_interval
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        visual_maps = []
        imgs = []
        for vi, data in enumerate(gen_tqdm, 0):
            if vi % test_interval == 0 or vi == len(sub_valset) - 1:
                frame_signal = 'match'
            else:
                frame_signal = 'skip'
            if frame_signal == 'match' or not args.skip_flag:
                img, label = data
                for i in range(len(label)):
                    for key, data in label[i].items():
                        if torch.is_tensor(data):
                            label[i][key] = data.cuda()
                img = img.cuda()
                with torch.no_grad():
                    b, c, h, w = img.shape
                    if h % 32 != 0:
                        pad_h = 32 - h % 32
                    else:
                        pad_h = 0
                    if w % 32 != 0:
                        pad_w = 32 - w % 32
                    else:
                        pad_w = 0
                    pad_dims = (0, pad_w, 0, pad_h)
                    img = F.pad(img, pad_dims, "constant")
                    h, w = img.size(2), img.size(3)
                    place_holder_img = torch.zeros((1, h, w)).cuda()
                    pre_map, gt_den, pre_share_map, gt_share_den, pre_in_out_map, gt_in_out_den, loss_dict = model(img, label)
                    pre_in_out_map[pre_in_out_map < 0] = 0
                    gt_count, pred_cnt = gt_den[0].sum().item(),  pre_map[0].sum().item()
                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                    sing_cnt_errors['mae'].update(s_mae)
                    sing_cnt_errors['mse'].update(s_mse)
                    if vi == 0:
                        pred_dict['first_frame'] = pred_cnt
                        gt_dict['first_frame'] = gt_count
                    pred_dict['inflow'].append(pre_in_out_map[1].sum().item())
                    pred_dict['outflow'].append(pre_in_out_map[0].sum().item())
                    gt_dict['inflow'].append(gt_in_out_den[1].sum().item())
                    gt_dict['outflow'].append(gt_in_out_den[0].sum().item())
                    if vi % test_interval == 0:
                        img0 = img[0]
                        gt_den0 = gt_den[0]
                        pre_map0 = pre_map[0]
                        if vi == 0:
                            gt_share_den_before = deepcopy(place_holder_img)
                            pre_share_den_before = deepcopy(place_holder_img)
                            gt_in_den = deepcopy(place_holder_img)
                            pre_in_den = deepcopy(place_holder_img)
                        else:
                            gt_share_den_before = previous_gt_share_den[1]
                            pre_share_den_before = previous_pre_share_map[1]
                            gt_in_den = previous_gt_in_out_den[1]
                            pre_in_den = previous_pre_in_out_map[1]
                        gt_share_den_next = gt_share_den[0]
                        pre_share_den_next = pre_share_map[0]
                        gt_out_den = gt_in_out_den[0]
                        pre_out_den = pre_in_out_map[0]
                        visual_map = torch.stack([gt_den0, pre_map0, gt_share_den_before, pre_share_den_before, gt_in_den, pre_in_den, gt_share_den_next, pre_share_den_next,
                                                  gt_out_den, pre_out_den], dim=0)
                        visual_maps.append(visual_map)
                        imgs.append(img0)
                        previous_gt_share_den = gt_share_den
                        previous_pre_share_map = pre_share_map
                        previous_gt_in_out_den = gt_in_out_den
                        previous_pre_in_out_map = pre_in_out_map
                        if (vi + test_interval) > (len(sub_valset) - 1):
                            visual_map = torch.stack([gt_den[1], pre_map[1], gt_share_den[1], pre_share_map[1], gt_in_out_den[1], pre_in_out_map[1], deepcopy(place_holder_img),
                                                      deepcopy(place_holder_img), deepcopy(place_holder_img), deepcopy(place_holder_img)], dim=0)
                            visual_maps.append(visual_map)
                            imgs.append(img[1])
        visual_maps = torch.stack(visual_maps, dim=0)
        save_test_visual(visual_maps, imgs, scene_name, restore_transform, args.output_dir, 0, 0)
        scenes_pred_dict['all'].append(pred_dict)
        scenes_gt_dict['all'].append(gt_dict)
        if scene_name in scene_label:
            scene_l = scene_label[scene_name]
            if scene_l[0] == 0:
                scenes_pred_dict['density0'].append(pred_dict)
                scenes_gt_dict['density0'].append(gt_dict)
            if scene_l[0] == 1:
                scenes_pred_dict['density1'].append(pred_dict)
                scenes_gt_dict['density1'].append(gt_dict)
            if scene_l[0] == 2:
                scenes_pred_dict['density2'].append(pred_dict)
                scenes_gt_dict['density2'].append(gt_dict)
            if scene_l[0] == 3:
                scenes_pred_dict['density3'].append(pred_dict)
                scenes_gt_dict['density3'].append(gt_dict)
    for key in scenes_pred_dict.keys():
        s_pred_dict = scenes_pred_dict[key]
        s_gt_dict = scenes_gt_dict[key]
        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(s_pred_dict, s_gt_dict, intervals)
        if key == 'all':
            save_cnt_result = cnt_result
        print('MAE: %.2f, MSE: %.2f, WRAE: %.2f, WIAE: %.2f, WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
    print('Pre vs GT:', save_cnt_result)
    mae = sing_cnt_errors['mae'].avg
    mse = np.sqrt(sing_cnt_errors['mse'].avg)
    print('MAE: {:.2f}, MSE: {:.2f}'.format(mae, mse))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='MovingDroneCrowd')
    parser.add_argument('--output_dir', type=str, default='saved_drone')
    parser.add_argument('--test_intervals', type=int, default=4)
    parser.add_argument('--skip_flag', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--ckpt_dir', type=str, default='saved_drone/ep_1_iter_1750_mae_27.272_mse_31.772_seq_MAE_88.285_WRAE_83.176_MIAE_6.878_MOAE_7.274.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    setup_seed()
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    test(datasetting.cfg_data)