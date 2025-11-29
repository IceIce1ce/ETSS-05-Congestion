import warnings
warnings.filterwarnings("ignore")
from torch import optim
from model.video_crowd_flux import DAANet
from model.loss import ComputeKPILoss
from misc.utils import adjust_learning_rate, AverageMeter, update_model, print_NWPU_summary_det, save_results_mask
from tqdm import tqdm
from misc.gt_generate import GenerateGT
from model.points_from_den import get_ROI_and_MatchInfo
import os
import random
import numpy as np
import torch
import datasets
import argparse
import torch.nn.functional as F

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

class Trainer():
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.args = args
        if args.resume_path != '':
            self.resume = True
        else:
            self.resume = False
        # model
        self.net = DAANet(args).cuda()
        # optimizer
        params = [{"params": self.net.Extractor.parameters(), 'lr': args.lr_base, 'weight_decay': args.weight_decay},
                  {"params": self.net.deformable_alignment.parameters(), "lr": args.lr_thre, 'weight_decay': args.weight_decay},
                  {"params": self.net.mask_predict_layer.parameters(), "lr": args.lr_thre, 'weight_decay': args.weight_decay},
                  {"params": self.net.ASAM.parameters(), "lr": args.lr_base, 'weight_decay': args.weight_decay}]
        self.optimizer = optim.Adam(params)
        self.i_tb = 0
        self.epoch = 1
        self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20, 'seq_MSE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20}
        # resume training
        if self.args.resume_path != '':
            print('Load ckpt from:', self.args.resume_path)
            # self.optimizer = optim.Adam(params)
            latest_state = torch.load(self.args.resume_path, map_location='cuda')
            self.train_record = latest_state['train_record']
            self.net.load_state_dict(latest_state['net'], strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch']
            self.i_tb = latest_state['i_tb']
            self.output_dir = latest_state['output_dir']
            self.args = latest_state['args']
        # train and test loader
        self.train_loader, self.val_loader, self.restore_transform = datasets.loading_data(self.args)
        # loss
        self.compute_kpi_loss = ComputeKPILoss(self, args)
        self.generate_gt = GenerateGT(args)
        self.feature_scale = args.feature_scale
        self.get_ROI_and_MatchInfo = get_ROI_and_MatchInfo(self.args.train_size, self.args.roi_radius)

    def forward(self):
        for epoch in range(self.epoch, self.args.epochs):
            self.epoch = epoch
            self.train()

    def train(self):
        self.net.train()
        lr1, lr2 = adjust_learning_rate(self.optimizer, self.epoch, self.args.lr_base, self.args.lr_thre, self.args.lr_decay)
        batch_loss = {'den': AverageMeter(), 'in': AverageMeter(), 'out': AverageMeter(), 'mask': AverageMeter(), 'con': AverageMeter(), 'scale_mask': AverageMeter(),
                      'scale_den': AverageMeter(), 'scale_io': AverageMeter()}
        for i, data in enumerate(self.train_loader, 0):
            self.i_tb += 1
            img, target = data # [4], [4]
            img = torch.stack(img, 0).cuda() # [4, 3, 768, 1024]
            img_pair_num = img.size(0) // 2
            # [4, 1, 768, 1024], [4, 1, 768, 1024], [4, 1, 768, 1024], [2, 1, 768, 1024], [2, 1, 768, 1024], [4, 3, 768, 1024], [2, 72, 768, 1024], [2, 72, 768, 1024], [2, 384, 192,356], [2, 384, 192,356]
            den_scales, final_den, mask, out_den, in_den, attns, f_flow, b_flow, feature1, feature2 = self.net(img)
            pre_inf_cnt, pre_out_cnt = in_den.sum(), out_den.sum()
            target_ratio = den_scales[0].shape[2] / img.shape[2] # 1.0
            for b in range(len(target)):        
                for key,data in target[b].items():
                    if torch.is_tensor(data):
                        target[b][key] = data.cuda()
            gt_den_scales = self.generate_gt.get_den(den_scales[0].shape, target, target_ratio, scale_num=len(den_scales)) # [4, 1, 768, 1024]
            gt_io_map = torch.zeros(img_pair_num, 2, den_scales[0].size(2), den_scales[0].size(3)).cuda() # [2, 2, 768, 1024]
            gt_inflow_cnt = torch.zeros(img_pair_num).cuda() # [2]
            gt_outflow_cnt = torch.zeros(img_pair_num).cuda() # [2]
            con_loss = torch.Tensor([0]).cuda()
            for pair_idx in range(img_pair_num):
                count_in_pair = [target[pair_idx * 2]['points'].size(0), target[pair_idx * 2 + 1]['points'].size(0)]
                if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                    match_gt, pois = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2 + 1],'ab') # match_gt: [8, 2], [6], [1], poins: [23, 5]
                    # [2, 2, 768, 1024], [2], [2]
                    gt_io_map, gt_inflow_cnt, gt_outflow_cnt = self.generate_gt.get_pair_io_map(pair_idx, target, match_gt, gt_io_map, gt_outflow_cnt, gt_inflow_cnt, target_ratio)
                    if len(match_gt['a2b'][:, 0]) > 0:
                        con_loss = con_loss + self.compute_kpi_loss.compute_con_loss(pair_idx, feature1, feature2, match_gt, pois, count_in_pair, self.feature_scale)
            con_loss /= args.train_batch_size
            gt_mask = (gt_io_map > 0).float() # [2, 2, 768, 1024]
            kpi_loss = self.compute_kpi_loss(final_den, den_scales, gt_den_scales, mask, gt_mask,  out_den, in_den, gt_io_map, pre_inf_cnt, pre_out_cnt, gt_inflow_cnt, gt_outflow_cnt, attns)
            all_loss = (kpi_loss + con_loss * args.con_weight).sum()
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()
            batch_loss['den'].update(self.compute_kpi_loss.cnt_loss.sum().item())
            batch_loss['in'].update(self.compute_kpi_loss.in_loss.sum().item())
            batch_loss['out'].update(self.compute_kpi_loss.out_loss.sum().item())
            batch_loss['mask'].update(self.compute_kpi_loss.mask_loss.sum().item())
            batch_loss['scale_den'].update(self.compute_kpi_loss.cnt_loss_scales.sum().item())
            batch_loss['con'].update(con_loss.item())
            if self.i_tb % self.args.print_freq == 0:
                print('Epoch: {}, Iter: [{}/{}], loss_den_overall: {:.4f}, loss_den: {:.4f}, loss_mask: {:.4f}, loss_in: {:.4f}, loss_out: {:.4f}, loss_con: {:.4f}, Base LR: {:.8f}, '
                      'Thre LR: {:.8f}'.format(self.epoch, self.i_tb, len(self.train_loader), batch_loss['den'].avg, batch_loss['scale_den'].avg, batch_loss['mask'].avg,
                       batch_loss['in'].avg, batch_loss['out'].avg, batch_loss['con'].avg, lr1, lr2))
            if self.i_tb % self.args.save_vis_freq == 0:
                save_results_mask(self.args, None, self.i_tb, self.restore_transform, 0, img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0),
                                  final_den[0].detach().cpu().numpy(), final_den[1].detach().cpu().numpy(), out_den[0].detach().cpu().numpy(), in_den[0].detach().cpu().numpy(),
                                  gt_io_map[0].unsqueeze(0).detach().cpu().numpy(), (attns[0, :, :, :]).unsqueeze(0).detach().cpu().numpy(), (attns[1, :, :, :]).unsqueeze(0).detach().cpu().numpy(),
                                  f_flow, b_flow, den_scales, gt_den_scales, mask, gt_mask)
            if (self.i_tb % self.args.val_freq == 0) and  (self.i_tb > self.args.val_start):
                self.validate()
                self.net.train()

    def validate(self):
        with torch.no_grad():
            self.net.eval()
            sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
            scenes_pred_dict = []
            scenes_gt_dict = []
            for scene_id, sub_valset in  enumerate(self.val_loader, 0):
                gen_tqdm = tqdm(sub_valset)
                video_time = len(sub_valset) + self.args.val_intervals
                pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
                gt_dict  = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
                for vi, data in enumerate(gen_tqdm, 0):
                    img, target = data
                    img, target = img[0], target[0]
                    img = torch.stack(img, 0).cuda() # [2, 3, 1088, 1920]
                    img_pair_num = img.shape[0] // 2
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
                    if vi % self.args.val_intervals == 0 or vi == len(sub_valset) - 1:
                        frame_signal = 'match'
                    else:
                        frame_signal = 'skip'
                    if frame_signal == 'skip':
                        continue
                    else:
                        den_scales, final_den, _, out_den, in_den, _, _, _, _, _, = self.net(img) # [2, 1, 1088, 1920], [2, 1, 1088, 1920], [1, 1, 1088, 1920], [1, 1, 1088, 1920]
                        pre_inf_cnt, pre_out_cnt = in_den.sum().detach().cpu(), out_den.sum().detach().cpu()
                        target_ratio = final_den.shape[2] / img.shape[2] # 1.0
                        for b in range(len(target)):
                            target[b]["points"] = target[b]["points"] * target_ratio
                            target[b]["sigma"] = target[b]["sigma"] * target_ratio
                            for key,data in target[b].items():
                                if torch.is_tensor(data):
                                    target[b][key] = data.cuda()
                        gt_den_scales = self.generate_gt.get_den(den_scales[0].shape, target, target_ratio, scale_num=len(den_scales)) # [2, 1, 1088, 1920]
                        gt_den = gt_den_scales[0]
                        assert final_den.size() == gt_den.size()
                        gt_io_map = torch.zeros(img_pair_num, 2, den_scales[0].size(2), den_scales[0].size(3)).cuda() # [1, 2, 1088, 1920]
                        gt_in_cnt = torch.zeros(img_pair_num).detach() # [1]
                        gt_out_cnt = torch.zeros(img_pair_num).detach() # [1]
                        for pair_idx in range(img_pair_num):
                            count_in_pair = [target[pair_idx * 2]['points'].size(0), target[pair_idx * 2 + 1]['points'].size(0)]
                            if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                                match_gt, pois = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab') # match_gt: [38, 2], [1], [3], pois: [80, 5]
                                gt_io_map, gt_in_cnt, gt_out_cnt = self.generate_gt.get_pair_io_map(pair_idx, target, match_gt, gt_io_map, gt_out_cnt, gt_in_cnt, target_ratio)
                        gt_count, pred_cnt = gt_den[0].sum().item(),  final_den[0].sum().item()
                        s_mae = abs(gt_count - pred_cnt)
                        s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                        sing_cnt_errors['mae'].update(s_mae)
                        sing_cnt_errors['mse'].update(s_mse)
                        if vi == 0:
                            pred_dict['first_frame'] = final_den[0].sum().item()
                            gt_dict['first_frame'] = len(target[0]['person_id'])
                        pred_dict['inflow'].append(pre_inf_cnt)
                        pred_dict['outflow'].append(pre_out_cnt)
                        gt_dict['inflow'].append(torch.tensor(gt_in_cnt))
                        gt_dict['outflow'].append(torch.tensor(gt_out_cnt))
                        pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict,1) # [1], [1]
                        print('Den GT: {:.2f}, Den Pred: {:.2f}, MAE: {:.2f}'.format(gt_count, pred_cnt, s_mae))
                        print('GT crowd flow: {}, GT inflow: {}'.format(gt_crowdflow_cnt.cpu().numpy()[0], gt_in_cnt.cpu().numpy()[0]))
                        print('Pred crowd flow: {:.2f}, Pred inflow: {:.2f}'.format(pre_crowdflow_cnt.cpu().numpy(), pre_inf_cnt.cpu().numpy()))
                scenes_pred_dict.append(pred_dict)
                scenes_gt_dict.append(gt_dict)
            MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, 1)
            print('MAE: {:.2f}, MSE: {:.2f}, WRAE: {:.2f}, MIAE: {:.2f}, MOAE: {:.2f}'.format(MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
            print('Pred vs GT:', cnt_result)
            mae = sing_cnt_errors['mae'].avg
            mse = np.sqrt(sing_cnt_errors['mse'].avg)
            self.train_record = update_model(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'seq_MSE': MSE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE}, val=True)
            print_NWPU_summary_det(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'seq_MSE': MSE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE})

def compute_metrics_single_scene(pre_dict, gt_dict, intervals, target=True):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt = torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2) # [1, 2], [1, 2]
    pre_crowdflow_cnt = pre_dict['first_frame']
    if target:
        gt_crowdflow_cnt = gt_dict['first_frame']
        all_data = zip(pre_dict['inflow'], pre_dict['outflow'], gt_dict['inflow'], gt_dict['outflow'])
    else:
        all_data = zip(pre_dict['inflow'], pre_dict['outflow'])
    for idx, data in enumerate(all_data,0):
        inflow_cnt[idx, 0] = data[0]
        outflow_cnt[idx, 0] = data[1]
        if target:
            inflow_cnt[idx, 1] = data[2]
            outflow_cnt[idx, 1] = data[3]
        if idx % intervals == 0 or idx == len(pre_dict['inflow']) - 1:
            pre_crowdflow_cnt += data[0]
            if target:
                gt_crowdflow_cnt += data[2]
    if target:
        return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt
    else:
        return pre_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals, target=True):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE': torch.zeros(scene_cnt, 2), 'WRAE': torch.zeros(scene_cnt, 2), 'MIAE': torch.zeros(0), 'MOAE': torch.zeros(0)}
    for i, (pre_dict, gt_dict) in enumerate(zip(scenes_pred_dict, scene_gt_dict), 0):
        time = pre_dict['time']
        if target:
            pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt = compute_metrics_single_scene(pre_dict, gt_dict,intervals,target)
        else:
            pre_crowdflow_cnt, inflow_cnt, outflow_cnt = compute_metrics_single_scene(pre_dict, gt_dict,intervals,target)
        if 'total_flow' in scene_gt_dict[0].keys():
            gt_crowdflow_cnt = scene_gt_dict[0]['total_flow'][i]
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae / (gt_crowdflow_cnt + 1e-10), time])
        if target:
            metrics['MIAE'] =  torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:,0] - inflow_cnt[:,1])])
            metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])
    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:, 0] * (metrics['WRAE'][:, 1] / (metrics['WRAE'][:, 1].sum() + 1e-10))) * 100
    if target:
        MIAE = torch.mean(metrics['MIAE'] )
        MOAE = torch.mean(metrics['MOAE'])
        return MAE,MSE, WRAE,MIAE,MOAE,metrics['MAE']
    return MAE, MSE, WRAE, metrics['MAE']

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='SENSE', choices=['SENSE', 'HT21', 'CARLA'])
    parser.add_argument('--output_dir', type=str, default='saved_sense')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=3035)
    # training config
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_vis_freq', type=int, default=500)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_base', type=float, default=5e-5) # density branch
    parser.add_argument('--lr_thre', type=float, default=1e-4) # mask branch
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epoch', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_size', type=int, nargs='+', default=[768, 1024])
    parser.add_argument('--train_frame_intervals', type=int, nargs='+', default=[40, 85])
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--mean_std', type=tuple, default=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    # model config
    parser.add_argument('--backbone', type=str, default='vgg')
    parser.add_argument('--con_weight', type=float, default=0.5)
    parser.add_argument('--scale_weight', type=float, nargs='+', default=[2,0.1,0.01])
    parser.add_argument('--cnt_weight', type=float, default=10)
    parser.add_argument('--mask_weight', type=float, default=1)
    parser.add_argument('--io_weight', type=float, default=1)
    parser.add_argument('--roi_radius', type=float, default=4.)
    parser.add_argument('--feature_scale', type=float, default=1 / 4.)
    parser.add_argument('--gaussian_sigma', type=float, default=4)
    parser.add_argument('--conf_block_size', type=int, default=16)
    parser.add_argument('--crop_rate', type=float, nargs='+', default=[0.8, 1.2])
    parser.add_argument('--den_factor', type=float, default=200.)
    # testing config
    parser.add_argument('--val_freq', type=int, default=2000)
    parser.add_argument('--val_start', type=int, default=0)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--val_intervals', type=int, default=50)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    if args.type_dataset == "SENSE":
        args.train_frame_intervals = [5,25]
        args.val_intervals = 10
    elif args.type_dataset == "CARLA":
        args.crop_rate = [0.6, 1.2]
        args.lr_base = 1e-5
        args.val_intervals = 62
    args.mode = 'train'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    setup_seed(args.seed)
    cc_trainer = Trainer(args)
    cc_trainer.forward()