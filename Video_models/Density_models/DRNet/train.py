import warnings
warnings.filterwarnings('ignore')
from torch import optim
from misc.utils import adjust_learning_rate, AverageMeter, save_results_more, make_matching_plot_fast, update_model, print_NWPU_summary_det
import cv2
from model.VIC import Video_Individual_Counter
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.cm as cm
from pathlib import Path
from misc.KPI_pool import Task_KPI_Pool
import random
import numpy as np
import torch
import datasets
from config import cfg
from importlib import import_module
import argparse

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

class Trainer():
    def __init__(self, cfg_data, args):
        self.args = args
        # model
        self.net = Video_Individual_Counter(cfg, cfg_data)
        # train and val loader
        self.train_loader, self.val_loader, self.restore_transform = datasets.loading_data(args.type_dataset, args.val_intervals)
        # optimizer
        params = [{"params": self.net.Extractor.parameters(), 'lr': cfg.LR_Base, 'weight_decay': cfg.WEIGHT_DECAY},
                  {"params": self.net.Matching_Layer.parameters(), "lr": cfg.LR_Thre, 'weight_decay': cfg.WEIGHT_DECAY}]
        self.optimizer = optim.Adam(params)
        self.i_tb = 0
        self.epoch = 1
        self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE': 1e20, 'WRAE': 1e20, 'MIAE': 1e20, 'MOAE': 1e20}
        if args.resume:
            latest_state = torch.load(args.resume_dir)
            self.net.load_state_dict(latest_state['net'], strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch']
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            print('Load ckpt from:', args.resume_dir)
        self.task_KPI = Task_KPI_Pool(task_setting={'den': ['gt_cnt', 'mae'], 'match': ['gt_pairs', 'pre_pairs']}, maximum_sample=args.maximum_sample)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            self.train()

    def train(self):
        self.net.train()
        lr1, lr2 = adjust_learning_rate(self.optimizer, self.epoch, cfg.LR_Base, cfg.LR_Thre, cfg.LR_DECAY)
        batch_loss = {'match': AverageMeter(), 'den': AverageMeter(), 'hard': AverageMeter(), 'norm': AverageMeter()}
        for i, data in enumerate(self.train_loader, 0):
            self.i_tb += 1
            img, label = data # [3, 768, 1024]
            pre_map, gt_map, correct_pairs_cnt, match_pairs_cnt, TP, matched_results = self.net(img, label) # [8, 1, 768, 1024], [8, 1, 768, 1024], [1], [1]
            counting_mse_loss, matching_loss, hard_loss, norm_loss = self.net.loss
            pre_cnt = pre_map.sum()
            gt_cnt = gt_map.sum()
            self.task_KPI.add({'den': {'gt_cnt': gt_map.sum(), 'mae': max(0, gt_cnt - (gt_cnt - pre_cnt).abs())}, 'match': {'gt_pairs': match_pairs_cnt, 'pre_pairs': correct_pairs_cnt}})
            self.KPI = self.task_KPI.query()
            loss = torch.stack([counting_mse_loss, matching_loss + hard_loss])
            weight = torch.stack([self.KPI['den'], self.KPI['match']]).to(loss.device) # [2]
            weight = -(1 - weight) * torch.log(weight + 1e-8)
            self.weight = weight / weight.sum()
            all_loss = (self.weight * loss).sum()
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()
            batch_loss['match'].update(matching_loss.item())
            batch_loss['den'].update(counting_mse_loss.item())
            batch_loss['hard'].update(hard_loss.item())
            batch_loss['norm'].update(norm_loss.item())
            if self.i_tb % cfg.PRINT_FREQ == 0:
                print('Epoch: {}, Iter: [{}/{}], Loss reg: {:.4f}, Loss match: {:.2f}, Loss hard: {:.4f}, acc_d: {:.2f}, acc_m: {:.2f}'.format(self.epoch, self.i_tb,
                       len(self.train_loader), batch_loss['den'].avg, batch_loss['match'].avg,batch_loss['hard'].avg, self.KPI['den'].item(), self.KPI['match'].item()))
                print('GT: {:.2f}, Pred: {:.2f}'.format(gt_cnt.item(), pre_cnt.item()))
            if self.i_tb % 400 == 0:
                kpts0 = label[0]['points'].cpu().numpy() # [35, 2]
                kpts1 = label[1]['points'].cpu().numpy() # [34, 2]
                id0 = label[0]['person_id'].cpu().numpy() # [35]
                id1 = label[1]['person_id'].cpu().numpy() # [34]
                matches = matched_results['matches0'][0].cpu().detach().numpy() # [35]
                confidence = matched_results['matching_scores0'][0].cpu().detach().numpy() # [35]
                if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                    save_visImg(kpts0, kpts1, matches, confidence, self.i_tb, img[0].clone(), img[1].clone(), 1, self.args.output_dir, id0, id1, scene_id='',
                                restore_transform=self.restore_transform)
                save_results_more(self.i_tb, self.args.output_dir, self.restore_transform, img[1].clone().unsqueeze(0), pre_map[1].detach().cpu().numpy(),
                                  gt_map[1].detach().cpu().numpy(), pre_map[1].detach().cpu().numpy(), pre_map[1].detach().cpu().numpy(), pre_map[1].detach().cpu().numpy())
            if self.i_tb % 2500 == 0:
                self.validate()
                self.net.train()

    def validate(self):
        self.net.eval()
        sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
        scenes_pred_dict = []
        scenes_gt_dict = []
        for scene_id, sub_valset in  enumerate(self.val_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + self.args.val_intervals
            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict  = {'id': scene_id, 'time' :video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            for vi, data in enumerate(gen_tqdm, 0):
                img, target = data
                img, target = img[0], target[0]
                img = torch.stack(img,0) # [2, 3, 1280, 720]
                with torch.no_grad():
                    b, c, h, w = img.shape
                    if h % 16 != 0:
                        pad_h = 16 - h % 16
                    else: pad_h = 0
                    if w % 16 != 0:
                        pad_w = 16 - w % 16
                    else: pad_w = 0
                    pad_dims = (0, pad_w, 0, pad_h)
                    img = F.pad(img, pad_dims, "constant") # [2, 3, 1280, 720]
                    if vi % self.args.val_intervals == 0 or vi == len(sub_valset) - 1:
                        frame_signal = 'match'
                    else:
                        frame_signal = 'skip'
                    if frame_signal == 'skip':
                        continue
                    else:
                        pred_map, gt_den, matched_results = self.net.val_forward(img, target) # [2, 1, 1280, 720], [2,1 , 1280, 720]
                        gt_count, pred_cnt = gt_den[0].sum().item(),  pred_map[0].sum().item()
                        s_mae = abs(gt_count - pred_cnt)
                        s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                        sing_cnt_errors['mae'].update(s_mae)
                        sing_cnt_errors['mse'].update(s_mse)
                        if vi == 0:
                            pred_dict['first_frame'] = pred_map[0].sum().item()
                            gt_dict['first_frame'] = len(target[0]['person_id'])
                        pred_dict['inflow'].append(matched_results['pre_inflow'])
                        pred_dict['outflow'].append(matched_results['pre_outflow'])
                        gt_dict['inflow'].append(matched_results['gt_inflow'])
                        gt_dict['outflow'].append(matched_results['gt_outflow'])
                        if frame_signal == 'match':
                            pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ =compute_metrics_single_scene(pred_dict, gt_dict,1)
                            print('Den gt: {:.2f}, Den Pred: {:.2f}, MAE: {:.2f}, GT crowd flow: {:.2f}, Pred crowd flow: {:.2f}, GT inflow: {:.2f}, Pred inflow: {:.2f}'.
                                  format(gt_count, pred_cnt, s_mae, gt_crowdflow_cnt, pre_crowdflow_cnt, matched_results['gt_inflow'], matched_results['pre_inflow']))
                            kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy() # [0, 2]
                            kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy() # [0, 2]
                            matches = matched_results['matches0'].cpu().numpy() # [1, 0]
                            confidence = matched_results['matching_scores0'].cpu().numpy() # [1, 0]
                            if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                                save_visImg(kpts0, kpts1, matches, confidence, vi, img[0].clone(),img[1].clone(), self.args.val_intervals, self.args.val_vis_dir, None,
                                            None, scene_id, self.restore_transform)
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)
        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict,scenes_gt_dict, 1)
        print('MAE: {:.2f}, MSE: {:.2f}, WRAE: {:.2f}, WIAE: {:.2f}, WOAE: {:.2f}'.format(MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
        print('Pre vs GT:', cnt_result)
        mae = sing_cnt_errors['mae'].avg
        mse = np.sqrt(sing_cnt_errors['mse'].avg)
        self.train_record = update_model(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE}, args)
        print_NWPU_summary_det(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE})

def save_visImg(kpts0, kpts1, matches, confidence, vi, last_frame, cur_frame, intervals, save_path, id0=None, id1=None,scene_id='', restore_transform=None):
    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1,2)
    mkpts1 = kpts1[matches[valid]].reshape(-1,2)
    color = cm.jet(confidence[valid])
    out, out_by_point = make_matching_plot_fast(last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, path=None, show_keypoints=True, restore_transform=restore_transform, id0=id0, id1=id1)
    if save_path is not None:
        Path(save_path).mkdir(exist_ok=True)
        stem = '{}_{}_{}_matches'.format(scene_id, vi , vi+ intervals)
        out_file = str(Path(save_path, stem + '.png'))
        print('Writing image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        out_file = str(Path(save_path, stem + '_vis.png'))
        cv2.imwrite(out_file, out_by_point)

def compute_metrics_single_scene(pre_dict, gt_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt = torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2) # [1, 2], [1, 2]
    pre_crowdflow_cnt = pre_dict['first_frame']
    gt_crowdflow_cnt =  gt_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'], pre_dict['outflow'], gt_dict['inflow'], gt_dict['outflow']), 0):
        inflow_cnt[idx, 0] = data[0]
        inflow_cnt[idx, 1] = data[2]
        outflow_cnt[idx, 0] = data[1]
        outflow_cnt[idx, 1] = data[3]
        if idx % intervals == 0 or idx== len(pre_dict['inflow']) - 1:
            pre_crowdflow_cnt += data[0]
            gt_crowdflow_cnt += data[2]
    return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE': torch.zeros(scene_cnt, 2), 'WRAE': torch.zeros(scene_cnt, 2), 'MIAE': torch.zeros(0), 'MOAE': torch.zeros(0)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']
        pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt= compute_metrics_single_scene(pre_dict, gt_dict, intervals)
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae / (gt_crowdflow_cnt + 1e-10), time])
        metrics['MIAE'] =  torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:, 0] - inflow_cnt[:, 1])])
        metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])
    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:, 0] * (metrics['WRAE'][:, 1] / (metrics['WRAE'][:, 1].sum() + 1e-10))) * 100
    MIAE = torch.mean(metrics['MIAE'] )
    MOAE = torch.mean(metrics['MOAE'])
    return MAE, MSE, WRAE, MIAE, MOAE, metrics['MAE']

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--maximum_sample', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='saved_sense')
    parser.add_argument('--val_vis_dir', type=str, default='saved_sense_test')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str, default='')
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    setup_seed(args.seed)
    if args.type_dataset == 'HT21':
        args.val_freq = 1
        args.val_dense_start = 2
        args.val_intervals = 50
    elif args.type_dataset == 'SENSE':
        args.val_freq = 1
        args.val_dense_start = 0
        args.val_intervals = 15
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    cc_trainer = Trainer(datasetting.cfg_data, args)
    cc_trainer.forward()