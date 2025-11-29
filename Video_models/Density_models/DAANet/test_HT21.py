import warnings
warnings.filterwarnings("ignore")
import datasets
from model.video_crowd_flux import DAANet
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from train import compute_metrics_single_scene, compute_metrics_all_scenes
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
        try:
            net.load_state_dict(state_dict, strict=True)
        except:
            net.load_state_dict(state_dict["net"], strict=True)
        print('Load ckpt from:', args.ckpt_dir)
        net.cuda()
        net.eval()
        scenes_pred_dict = []
        scenes_gt_dict = []
        gt_flow_cnt = [133, 737, 734, 1040, 321]
        scene_names = ['HT21-11', 'HT21-12', 'HT21-13', 'HT21-14', 'HT21-15']
        if args.skip_flag:
            intervals = 1
        else:
            intervals = args.test_intervals
        for scene_id, sub_valset in enumerate(test_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + args.test_intervals
            scene_name = scene_names[scene_id]
            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': [], 'total_flow': gt_flow_cnt}
            for vi, data in enumerate(gen_tqdm, 0):
                    img, _ = data
                    img = img[0]
                    img = torch.stack(img,0).cuda() # [2, 3, 1080, 1920]
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
                    if vi % args.val_intervals == 0 or vi ==len(sub_valset) - 1:
                        frame_signal = 'match'
                    else:
                        frame_signal = 'skip'
                    if frame_signal == 'skip':
                        continue
                    else:
                        den_scales, pred_map, _, out_den, in_den, _, _, _, _, _ = net(img) # [2, 1, 1088, 1920], [2, 1, 1088, 1920], [1, 1, 1088, 1920], [1, 1, 1088, 1920]
                        pre_inf_cnt, pre_out_cnt = in_den.sum().detach().cpu(), out_den.sum().detach().cpu() # [1], [1]
                        pred_cnt = pred_map[0].sum().item()
                        if vi == 0:
                            pred_dict['first_frame'] = pred_map[0].sum().item()
                        pred_dict['inflow'].append(pre_inf_cnt)
                        pred_dict['outflow'].append(pre_out_cnt)
                        pre_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, None, 1, target=False) # [1]
                        print('Den GT: {}, Den Pred: {:.2f}, MAE: {}'.format(None, pred_cnt, None))
                        print('Pred crowd flow: {:.2f}, Pred inflow: {:.2f}'.format(pre_crowdflow_cnt.squeeze().cpu().numpy(), pre_inf_cnt.squeeze().cpu().numpy()))
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)
        MAE, MSE, WRAE, crowdflow_cnt = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, intervals, target=False)
        print('MAE: {:.2f}, MSE: {:.2f}, WRAE: {:.2f}'.format(MAE.data, MSE.data, WRAE.data))
        print('Pre vs GT:', crowdflow_cnt)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='HT21')
    parser.add_argument('--output_dir', type=str, default='saved_ht21')
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--mean_std', type=tuple, default=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    # testing config
    parser.add_argument('--test_intervals', type=int, default=62)
    parser.add_argument('--skip_flag', type=bool, default=True)
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