import warnings
warnings.filterwarnings("ignore")
import datasets
from misc.utils import AverageMeter
from model.video_crowd_flux import DAANet
from model.points_from_den import get_ROI_and_MatchInfo
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from train import compute_metrics_single_scene, compute_metrics_all_scenes
from misc.gt_generate import GenerateGT
import numpy as np
import torch

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def test(args):
    with torch.no_grad():
        # model
        net = DAANet(args)
        # test loader
        test_loader, restore_transform = datasets.loading_testset(args, mode=args.mode)
        state_dict = torch.load(args.ckpt_dir, map_location='cuda')
        net.load_state_dict(state_dict, strict=True)
        print('Load ckpt from:', args.ckpt_dir)
        net.cuda()
        net.eval()
        scenes_pred_dict = []
        gt_flow_cnt = [232, 204, 278, 82, 349]
        scene_names = ['11', '12', '13', '14', '15']
        generate_gt = GenerateGT(args)
        get_roi_and_matchinfo = get_ROI_and_MatchInfo(args.train_size, args.roi_radius)
        sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
        scenes_gt_dict = []
        intervals = 1
        for scene_id, sub_valset in enumerate(test_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + args.test_intervals
            scene_name = scene_names[scene_id]
            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': [], 'total_flow': gt_flow_cnt}
            for vi, data in enumerate(gen_tqdm, 0):
                img, target = data
                img, target = img[0], target[0]
                img = torch.stack(img, 0).cuda()
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
                img = F.pad(img, pad_dims, "constant")
                img_pair_num = img.shape[0] // 2
                if vi % args.test_intervals == 0 or vi == len(sub_valset) - 1:
                    frame_signal = 'match'
                else:
                    frame_signal = 'skip'
                if frame_signal == 'skip':
                    continue
                else:
                    den_scales, pred_map, _, out_den, in_den, _, _, _, _, _ = net(img)
                    pre_inflow, pre_outflow = in_den.sum().detach().cpu(), out_den.sum().detach().cpu()
                    target_ratio = pred_map.shape[2] / img.shape[2]
                    for b in range(len(target)):
                        target[b]["points"] = target[b]["points"] * target_ratio
                        target[b]["sigma"] = target[b]["sigma"] * target_ratio
                        for key, data in target[b].items():
                            if torch.is_tensor(data):
                                target[b][key] = data.cuda()
                    gt_den_scales = generate_gt.get_den(den_scales[0].shape, target, target_ratio, scale_num=len(den_scales))
                    gt_den = gt_den_scales[0]
                    assert pred_map.size() == gt_den.size()
                    gt_io_map = torch.zeros(img_pair_num, 4, den_scales[0].size(2), den_scales[0].size(3)).cuda()
                    gt_in_cnt = torch.zeros(img_pair_num).detach()
                    gt_out_cnt = torch.zeros(img_pair_num).detach()
                    assert pred_map.size() == gt_den.size()
                    for pair_idx in range(img_pair_num):
                        count_in_pair = [target[pair_idx * 2]['points'].size(0), target[pair_idx * 2 + 1]['points'].size(0)]
                        if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                            match_gt, _ = get_roi_and_matchinfo(target[pair_idx * 2], target[pair_idx * 2 + 1], 'ab')
                            gt_io_map, gt_in_cnt, gt_out_cnt = generate_gt.get_pair_io_map(pair_idx, target, match_gt, gt_io_map, gt_out_cnt, gt_in_cnt, target_ratio)
                    gt_count, pred_cnt = gt_den[0].sum().item(), pred_map[0].sum().item()
                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                    sing_cnt_errors['mae'].update(s_mae)
                    sing_cnt_errors['mse'].update(s_mse)
                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                        gt_dict['first_frame'] = len(target[0]['person_id'])
                    pred_dict['inflow'].append(pre_inflow)
                    pred_dict['outflow'].append(pre_outflow)
                    gt_dict['inflow'].append(torch.tensor(gt_in_cnt).clone().detach())
                    gt_dict['outflow'].append(torch.tensor(gt_out_cnt).clone().detach())
                    pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict, 1)
                    print('Den GT: {}, Den Pred: {:.2f}, MAE: {}'.format(gt_count, pred_cnt, s_mae))
                    print('GT crowd flow: {}, GT inflow: {}'.format(gt_crowdflow_cnt.cpu().numpy(), gt_in_cnt.cpu().numpy()))
                    print('Pred crowd flow: {}, Pred inflow: {}'.format(pre_crowdflow_cnt.cpu().numpy(), pre_inflow.cpu().numpy()))
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)
        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, intervals)
        mae = sing_cnt_errors['mae'].avg
        mse = np.sqrt(sing_cnt_errors['mse'].avg)
        print('Den MAE: {:.2f}, Den MSE: {:.2f}, MAE: {:.2f}, MSE: {:.2f}, WRAE: {:.2f}, MIAE: {:.2f}, MOAE: {:.2f}'.format(mae, mse, MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
        print('Pre vs GT:', cnt_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='CARLA')
    parser.add_argument('--output_dir', type=str, default='saved_carla')
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--mean_std', type=tuple, default=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    # testing config
    parser.add_argument('--train_size', type=int, nargs='+', default=[768, 1024])
    parser.add_argument('--test_intervals', type=int, default=62)
    parser.add_argument('--val_batch_size', type=int, default=1)
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
