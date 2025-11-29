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
from train import compute_metrics_single_scene,compute_metrics_all_scenes
import  os.path as osp
from model.MatchTool.compute_metric import associate_pred2gt_point_vis
import os
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
    with open(osp.join(cfg_data.DATA_PATH, 'scene_label.txt'), 'r') as f:
        lines = f.readlines()
    scene_label = {}
    for line in lines:
        line = line.rstrip().split(' ')
        scene_label.update({line[0]: [int(i) for i in line[1:]]})
    # test loader
    test_loader, restore_transform = datasets.loading_testset(args.type_dataset, test_interval=args.test_intervals, mode='test')
    state_dict = torch.load(args.ckpt_dir)
    net.load_state_dict(state_dict, strict=True)
    print('Load ckpt from:', args.ckpt_dir)
    net.eval()
    scenes_pred_dict = {'all': [], 'in': [], 'out': [], 'day': [], 'night': [], 'scenic0': [], 'scenic1': [],'scenic2': [], 'scenic3': [], 'scenic4': [], 'scenic5': [],
                        'density0': [], 'density1': [], 'density2': [], 'density3': [], 'density4': []}
    scenes_gt_dict = {'all': [], 'in': [], 'out': [], 'day': [], 'night': [], 'scenic0': [], 'scenic1': [], 'scenic2': [], 'scenic3': [], 'scenic4': [], 'scenic5': [],
                      'density0': [], 'density1' :[], 'density2': [], 'density3': [], 'density4': []}
    if args.skip_flag:
        intervals = 1
    else:
        intervals = args.test_intervals
    for scene_id, sub_valset in enumerate(test_loader, 0):
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + args.test_intervals
        scene_name = ''
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        for vi, data in enumerate(gen_tqdm, 0):
            img, target = data
            img, target = img[0], target[0]
            scene_name = target[0]['scene_name']
            img = torch.stack(img, 0) # [2, 3, 720, 1280]
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
                img = F.pad(img, pad_dims, "constant") # [2, 3, 720, 1280]
                if vi % args.test_intervals == 0 or vi == len(sub_valset) - 1:
                    frame_signal = 'match'
                else:
                    frame_signal = 'skip'
                if frame_signal == 'match' or not args.skip_flag:
                    pred_map, gt_den, matched_results = net.val_forward(img, target) # [2, 1, 720, 1280], [2, 1, 720, 1280]
                    gt_count, pred_cnt = gt_den[0].sum().item(), pred_map[0].sum().item()
                    pred_cnt = pred_map[0].sum().item()
                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                        gt_dict['first_frame'] = len(target[0]['person_id'])
                    pred_dict['inflow'].append(matched_results['pre_inflow'])
                    pred_dict['outflow'].append(matched_results['pre_outflow'])
                    gt_dict['inflow'].append(matched_results['gt_inflow'])
                    gt_dict['outflow'].append(matched_results['gt_outflow'])
                if frame_signal == 'match':
                    pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict, intervals)
                    print('Den gt: {:.2f}, Den pred: {:.2f}, Gt crowd flow: {:.2f}, Pred crowd flow: {:.2f}, GT inflow: {:.2f}, Pred inflow: {:.2f}'.
                          format(gt_count, pred_cnt, gt_crowdflow_cnt, pre_crowdflow_cnt, matched_results['gt_inflow'], matched_results['pre_inflow']))
                    kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
                    kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()
                    matches = matched_results['matches0'].cpu().numpy()
                    confidence = matched_results['matching_scores0'].cpu().numpy()
                    if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                        save_visImg(kpts0, kpts1, matches, confidence, vi, img[0].clone(), img[1].clone(), args.test_intervals, osp.join(args.output_dir, scene_name),
                                    None, None, scene_name, restore_transform)
                        save_inflow_outflow_density(img, matched_results['scores'], matched_results['pre_points'], matched_results['target'], matched_results['match_gt'],
                                                    osp.join(args.output_dir,scene_name), scene_name, vi, args.test_intervals)
        scenes_pred_dict['all'].append(pred_dict)
        scenes_gt_dict['all'].append(gt_dict)
        scene_l = scene_label[scene_name]
        if scene_l[0] == 0:
            scenes_pred_dict['in'].append(pred_dict)
            scenes_gt_dict['in'].append(gt_dict)
        if scene_l[0] == 1:
            scenes_pred_dict['out'].append(pred_dict)
            scenes_gt_dict['out'].append(gt_dict)
        if scene_l[1] == 0:
            scenes_pred_dict['day'].append(pred_dict)
            scenes_gt_dict['day'].append(gt_dict)
        if scene_l[1] == 1:
            scenes_pred_dict['night'].append(pred_dict)
            scenes_gt_dict['night'].append(gt_dict)
        if scene_l[2] == 0:
            scenes_pred_dict['scenic0'].append(pred_dict)
            scenes_gt_dict['scenic0'].append(gt_dict)
        if scene_l[2] == 1:
            scenes_pred_dict['scenic1'].append(pred_dict)
            scenes_gt_dict['scenic1'].append(gt_dict)
        if scene_l[2] == 2:
            scenes_pred_dict['scenic2'].append(pred_dict)
            scenes_gt_dict['scenic2'].append(gt_dict)
        if scene_l[2] == 3:
            scenes_pred_dict['scenic3'].append(pred_dict)
            scenes_gt_dict['scenic3'].append(gt_dict)
        if scene_l[2] == 4:
            scenes_pred_dict['scenic4'].append(pred_dict)
            scenes_gt_dict['scenic4'].append(gt_dict)
        if scene_l[2] == 5:
            scenes_pred_dict['scenic5'].append(pred_dict)
            scenes_gt_dict['scenic5'].append(gt_dict)
        if scene_l[3] == 0:
            scenes_pred_dict['density0'].append(pred_dict)
            scenes_gt_dict['density0'].append(gt_dict)
        if scene_l[3] == 1:
            scenes_pred_dict['density1'].append(pred_dict)
            scenes_gt_dict['density1'].append(gt_dict)
        if scene_l[3] == 2:
            scenes_pred_dict['density2'].append(pred_dict)
            scenes_gt_dict['density2'].append(gt_dict)
        if scene_l[3] == 3:
            scenes_pred_dict['density3'].append(pred_dict)
            scenes_gt_dict['density3'].append(gt_dict)
        if scene_l[3] == 4:
            scenes_pred_dict['density4'].append(pred_dict)
            scenes_gt_dict['density4'].append(gt_dict)
    for key in scenes_pred_dict.keys():
        s_pred_dict = scenes_pred_dict[key]
        s_gt_dict = scenes_gt_dict[key]
        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(s_pred_dict, s_gt_dict, intervals)
        if key == 'all':
            save_cnt_result = cnt_result
        print('MAE: %.2f, MSE: %.2f,  WRAE: %.2f, WIAE: %.2f, WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
    print(save_cnt_result)

def save_visImg(kpts0, kpts1, matches, confidence, vi, last_frame, cur_frame, intervals, save_path, id0=None, id1=None, scene_id='', restore_transform=None):
    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1, 2)
    mkpts1 = kpts1[matches[valid]].reshape(-1, 2)
    color = cm.jet(confidence[valid])
    out, out_by_point = make_matching_plot_fast(last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, path=None, show_keypoints=True, restore_transform=restore_transform, id0=id0, id1=id1)
    if save_path is not None:
        os.makedirs(save_path, mode =0o777, exist_ok=True)
        stem = '{}_{}_{}_matches'.format(scene_id, vi, vi + intervals)
        out_file = str(Path(save_path, stem + '.png'))
        print('Writing image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        out_file = str(Path(save_path, stem + '_vis.png'))
        cv2.imwrite(out_file, out_by_point)

def generate_cycle_mask(height, width, back_color, fore_color):
    x, y = np.ogrid[-height:height + 1, -width:width + 1]
    cir_idx = ((x) ** 2 / (height ** 2) + (y) ** 2 / (width ** 2) <= 1)
    mask = np.zeros((2 * height + 1, 2 * width + 1, 3)).astype(np.uint8)
    mask[cir_idx == 0] = back_color
    mask[cir_idx == 1] = fore_color
    return mask

def save_inflow_outflow_density(img, scores, pre_points, target, match_gt, save_path, scene_id, vi, intervals):
    scores = scores.cpu().numpy()
    _, __, img_h, img_w = img.size()
    gt_inflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    gt_outflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    pre_inflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    pre_outflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    RoyalBlue1 = np.array([255, 118, 72])
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    gt_inflow[:, :, 0:3] = RoyalBlue1
    gt_outflow[:, :, 0:3] = RoyalBlue1
    pre_inflow[:, :, 0:3] = RoyalBlue1
    pre_outflow[:, :, 0:3] = RoyalBlue1
    kernel = 8
    wide = 2 * kernel + 1
    pre_outflow_p = pre_points[0][scores[:-1, -1] > 0.4][:, 2:4]
    tp_pred_index, fp_pred_index, tp_gt_index, fn_gt_index = associate_pred2gt_point_vis(pre_outflow_p, target[0], match_gt['un_a'].cpu().numpy())
    for row_id, pos in enumerate(pre_outflow_p, 0):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        if row_id in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red)
        if row_id not in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, green)
        pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h), max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    for pos in (target[0]['points'][match_gt['un_a']][fn_gt_index]):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, blue)
        pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h), max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    pre_inflow_p = pre_points[1][scores[-1, :-1] > 0.4][:, 2:4]
    tp_pred_index, fp_pred_index, tp_gt_index, fn_gt_index = associate_pred2gt_point_vis(pre_inflow_p, target[1], match_gt['un_b'].cpu().numpy())
    for column_id, pos in enumerate(pre_inflow_p, 0):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        if column_id in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red)
        if column_id not in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, green)
        pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h), max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    for pos in (target[1]['points'][match_gt['un_b']][fn_gt_index]):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, blue)
        pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h), max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    for row_id in match_gt['un_a'].cpu().numpy():
        w, h = target[0]['points'][row_id].cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, [0, 0, 255])
        gt_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h), max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    for column_id in match_gt['un_b'].cpu().numpy():
        w, h = target[1]['points'][column_id].cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, [0, 0, 255])
        gt_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h), max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    os.makedirs(save_path, mode=0o777, exist_ok=True)
    stem = '{}_{}_{}_matches_outflow_pre_{}'.format(scene_id, vi, vi + intervals, np.round(scores[:-1, -1].sum(), 2))
    out_file = str(Path(save_path, stem + '.png'))
    print('Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, pre_outflow)
    stem = '{}_{}_{}_matches_inflow_pre_{}'.format(scene_id, vi, vi + intervals, np.round(scores[-1, :-1].sum(), 2))
    out_file = str(Path(save_path, stem + '.png'))
    print('Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, pre_inflow)
    stem = '{}_{}_{}_matches_outflow_gt_{}'.format(scene_id, vi, vi + intervals, match_gt['un_a'].size(0))
    out_file = str(Path(save_path, stem + '.png'))
    print('Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, gt_outflow)
    stem = '{}_{}_{}_matches_inflow_gt_{}'.format(scene_id, vi, vi + intervals, match_gt['un_b'].size(0))
    out_file = str(Path(save_path, stem + '.png'))
    print('Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, gt_inflow)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument('--output_dir', type=str, default='saved_sense')
    parser.add_argument('--test_intervals', type=int, default=15)
    parser.add_argument('--skip_flag', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--ckpt_dir', type=str, default='saved_sense/ep_1_iter_2500_mae_4.028_mse_6.252_seq_MAE_13.645_WRAE_19.896_MIAE_2.405_MOAE_2.255.pth')
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    setup_seed(args.seed)
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    test(datasetting.cfg_data)