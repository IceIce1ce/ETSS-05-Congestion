import warnings
warnings.filterwarnings("ignore")
from torch import optim
from misc.utils import AverageMeter, adjust_learning_rate, save_results_color, update_model, print_NWPU_summary_det, save_results_mask
from model.video_crowd_count import video_crowd_count
from tqdm import tqdm
import torch.nn.functional as F
from misc.KPI_pool import Task_KPI_Pool
import os
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
        self.output_dir = args.output_dir
        # model
        self.net = video_crowd_count(cfg, cfg_data)
        # train and val loader
        self.train_loader, self.color_loader, self.val_loader, self.restore_transform = datasets.loading_data(args.type_dataset, args.VAL_INTERVALS)
        # optimizer
        params = [{"params": self.net.Extractor.parameters(), 'lr': cfg.LR_Base, 'weight_decay': cfg.WEIGHT_DECAY},
                  {"params": self.net.optical_defromable_layer.parameters(), "lr": cfg.LR_Thre, 'weight_decay': cfg.WEIGHT_DECAY},
                  {"params": self.net.mask_predict_layer.parameters(), "lr": cfg.LR_Thre, 'weight_decay': cfg.WEIGHT_DECAY}]
        self.optimizer = optim.Adam(params)
        self.i_tb = 0
        self.epoch = 1
        if args.task == "LAB":
            self.train_record = {'best_model_name': '', 'color_loss': 1e20, 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20}
        else:
            self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20}
        # resume training
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state, strict=True)
            print('Load ckpt from:', cfg.RESUME_PATH)
        self.task_KPI = Task_KPI_Pool(task_setting={'den': ['gt_cnt', 'pre_cnt'], 'mask': ['gt_cnt', 'acc_cnt']}, maximum_sample=1000)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            self.train()

    def train(self):
        self.net.train()
        lr1, lr2 = adjust_learning_rate(self.optimizer, self.epoch, cfg.LR_Base, cfg.LR_Thre, cfg.LR_DECAY)
        if self.args.task == "LAB":
            batch_loss = {'mask': AverageMeter()}
        else:
            batch_loss = {'in': AverageMeter(), 'den': AverageMeter(), 'out': AverageMeter(), 'mask': AverageMeter()}
        if cfg.continuous:
            loader = self.color_loader
        else:
            loader = self.train_loader
        for i, data in enumerate(loader, 0):
            self.i_tb += 1
            img, img_rgb, label = data
            img = torch.stack(img, 0).cuda() # [4, 3, 768, 1024]
            img_rgb = torch.stack(img_rgb, 0).cuda() # [4, 3, 768, 1024]
            if self.args.task == "LAB":
                res = []
                for j in range(1, img.size(0), 2):
                    r = np.array(self.restore_transform(img[j].detach().clone())) # [768, 1024, 3]
                    r = torch.from_numpy(r).permute(2, 0, 1) # [3, 768, 1024]
                    res.append(r)
                res = torch.stack(res, 0).cuda() # [2, 3, 768, 1024]
                color = self.net.colorization(img, label, img_rgb)
                a = F.interpolate(color[:, :256, :, :], scale_factor=4)
                b = F.interpolate(color[:, 256:, :, :],scale_factor=4)
                loss = F.cross_entropy(a, res[:, 1, :, :].long()) + F.cross_entropy(b, res[:, 2, :, :].long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss['mask'].update(loss.item())
                if self.i_tb % cfg.PRINT_FREQ == 0:
                    print('Epoch: {}, Iter: [{}/{}], Loss mask: {:.4f}'.format(self.epoch, self.i_tb, len(loader), batch_loss['mask'].avg))
                if self.i_tb % 800 == 0:
                    with torch.no_grad():
                        mask = torch.concat([a[0:1].argmax(dim=1), b[0:1].argmax(dim=1)], axis=0)
                        mask = mask.permute(1, 2, 0)
                        path_color = os.path.join(args.output_dir, 'color')
                        if not os.path.exists(path_color):
                            os.makedirs(path_color)
                        save_results_color(self.i_tb, path_color, self.restore_transform, img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0), mask.clone().cpu())
            else:
                # [4, 1, 768, 1024], [4, 1, 768, 1024], [4, 1, 768, 1024], [2, 2, 768, 1024], [2, 2], [2], [2], [2], [2, 2, 768, 1024], [2, 2, 768, 1024]
                den, gt_den, mask, gt_mask, pre_out_cnt, gt_out_cnt, pre_inf_cnt, gt_in_cnt, f_flow, b_flow = self.net(img, label)
                counting_mse_loss, mask_loss, out_loss, in_loss, con_loss, offset_loss = self.net.loss
                pre_cnt = den.sum()
                gt_cnt = gt_den.sum()
                self.task_KPI.add({'den': {'gt_cnt': gt_cnt, 'pre_cnt': max(0,gt_cnt - (pre_cnt - gt_cnt).abs())}, 'mask': {'gt_cnt' : gt_out_cnt.sum() + gt_in_cnt.sum(),
                                   'acc_cnt': max(0,gt_out_cnt.sum()+gt_in_cnt.sum() - (pre_inf_cnt - gt_in_cnt).abs().sum() - (pre_out_cnt - gt_out_cnt).abs().sum()) }})
                self.KPI = self.task_KPI.query() # [1], [1]
                loss = torch.stack([counting_mse_loss, out_loss + in_loss + mask_loss]) # [2]
                weight = torch.stack([self.KPI['den'], self.KPI['mask']]).to(loss.device) # [2]
                weight = -(1 - weight) * torch.log(weight + 1e-8) # [2]
                self.weight = weight / weight.sum()
                all_loss = (self.weight * loss + offset_loss * 0 + con_loss * 0.1).sum()
                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()
                batch_loss['in'].update(in_loss.item())
                batch_loss['den'].update(counting_mse_loss.item())
                batch_loss['out'].update(out_loss.item())
                batch_loss['mask'].update(mask_loss.item())
                if self.i_tb % cfg.PRINT_FREQ == 0:
                    print('Epoch: {}, Iter: [{}/{}], Loss reg: {:.4f}, Loss: {:.4f}, Loss in: {:.4f}, Loss out: {:.4f}'.format(self.epoch, self.i_tb, len(loader),
                          batch_loss['den'].avg, batch_loss['mask'].avg,batch_loss['in'].avg, batch_loss['out'].avg))
                if self.i_tb % 800 == 0:
                    save_results_mask(self.i_tb, args.output_dir, self.restore_transform, img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0), den[0].detach().cpu().numpy(),
                                      gt_den[0].detach().cpu().numpy(), den[1].detach().cpu().numpy(), gt_den[1].detach().cpu().numpy(),
                                      mask[0, :, :, :].detach().cpu().numpy(), gt_mask[0, 0:1, :, :].detach().cpu().numpy(),
                                      mask[img.size(0) // 2, :, :, :].detach().cpu().numpy(), gt_mask[0, 1:2, :, :].detach().cpu().numpy(),
                                      f_flow[0].permute(1, 2, 0).detach().cpu().numpy(), b_flow[0].permute(1, 2, 0).detach().cpu().numpy())
            if self.i_tb % 1000 == 0:
                self.validate()
                self.net.train()
                self.net.flownet.eval()

    def validate(self):
        self.net.eval()
        sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'color': AverageMeter()}
        scenes_pred_dict = []
        scenes_gt_dict = []
        for scene_id, sub_valset in  enumerate(self.val_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + args.VAL_INTERVALS
            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict  = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            for vi, data in enumerate(gen_tqdm, 0):
                img, img_rgb, target = data
                img, img_rgb, target = img[0], img_rgb[0], target[0]
                img = torch.stack(img,0).cuda() # [2, 3, 1080, 1920]
                img_rgb = torch.stack(img_rgb,0).cuda() # [2, 3, 1080, 1920]
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
                    if vi % args.VAL_INTERVALS == 0 or vi ==len(sub_valset) - 1:
                        frame_signal = 'match'
                    else:
                        frame_signal = 'skip'
                    if frame_signal == 'skip':
                        continue
                    elif self.args.task == "LAB":
                        res = np.array(self.restore_transform(img[1].detach().clone())) # [1088, 1920, 3]
                        res = torch.from_numpy(res).permute(2, 0, 1).unsqueeze(0).cuda() # [1, 3, 1088, 1920]
                        color = self.net.colorization(img, target, img_rgb) # [1, 512, 272, 480]
                        a = F.interpolate(color[:, :256, :, :], scale_factor=4) # [1, 256, 1088, 1920]
                        b = F.interpolate(color[:, 256:, :, :], scale_factor=4) # [1, 256, 1088, 1920]
                        loss = F.cross_entropy(a, res[:, 1, :, :].long()) + F.cross_entropy(b, res[:, 2, :, :].long())
                        sing_cnt_errors['color'].update(loss.item())
                    else:
                        # [2, 1, 1088, 1920], [2, 1, 1088, 1920], [2, 1, 1088, 1920], [1, 2, 1088, 1920], 994.1, [1], 994.1, [1]
                        den, gt_den, mask, gt_mask, pre_out_cnt, gt_out_cnt, pre_inf_cnt, gt_in_cnt = self.net.test_or_validate(img, target)
                        gt_count, pred_cnt = gt_den[0].sum().item(),  den[0].sum().item()
                        s_mae = abs(gt_count - pred_cnt)
                        s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                        sing_cnt_errors['mae'].update(s_mae)
                        sing_cnt_errors['mse'].update(s_mse)
                        if vi == 0:
                            pred_dict['first_frame'] = den[0].sum().item()
                            gt_dict['first_frame'] = len(target[0]['person_id'])
                        pred_dict['inflow'].append(pre_inf_cnt)
                        pred_dict['outflow'].append(pre_out_cnt)
                        gt_dict['inflow'].append(torch.tensor(gt_in_cnt))
                        gt_dict['outflow'].append(torch.tensor(gt_out_cnt))
                        if frame_signal == 'match':
                            pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict,1)
                            print('Den gt: %.2f, Den pred: %.2f, MAE: %.2f, GT crowd flow: %.2f, Pred crowd flow: %.2f, GT inflow: %.2f, Pred inflow: %.2f'
                                  % (gt_count, pred_cnt, s_mae, gt_crowdflow_cnt, pre_crowdflow_cnt, gt_in_cnt,pre_inf_cnt))
            if self.args.task != "LAB":
                scenes_pred_dict.append(pred_dict)
                scenes_gt_dict.append(gt_dict)
        if self.args.task != "LAB":
            MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, 1)
            print('MAE: %.2f, MSE: %.2f, WRAE: %.2f, WIAE: %.2f, WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
            print('Pre vs GT:', cnt_result)
            mae = sing_cnt_errors['mae'].avg
            mse = np.sqrt(sing_cnt_errors['mse'].avg)
            self.train_record = update_model(self,{'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE})
            print_NWPU_summary_det(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE})
        else:
            print('Color val loss: %.4f' % sing_cnt_errors['color'].avg)
            if sing_cnt_errors['color'].avg < self.train_record['color_loss']:
                self.train_record['color_loss'] = sing_cnt_errors['color'].avg
                snapshot_name = 'ep_%d_iter_%d_loss_%.3f' % (self.epoch, self.i_tb, self.train_record['color_loss'])
                self.train_record['best_model_name'] = snapshot_name
                torch.save(self.net.state_dict(), os.path.join(args.output_dir, snapshot_name + '.pth'))
            latest_state = {'train_record': self.train_record, 'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': self.epoch, 'i_tb': self.i_tb}
            torch.save(latest_state, os.path.join(args.output_dir, 'latest_state.pth'))

def compute_metrics_single_scene(pre_dict, gt_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt =torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2)
    pre_crowdflow_cnt  = pre_dict['first_frame']
    gt_crowdflow_cnt =  gt_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow'], gt_dict['inflow'], gt_dict['outflow']), 0):
        inflow_cnt[idx, 0] = data[0]
        inflow_cnt[idx, 1] = data[2]
        outflow_cnt[idx, 0] = data[1]
        outflow_cnt[idx, 1] = data[3]
        if idx % intervals == 0 or  idx== len(pre_dict['inflow']) - 1:
            pre_crowdflow_cnt += data[0]
            gt_crowdflow_cnt += data[2]
    return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE': torch.zeros(scene_cnt,2), 'WRAE': torch.zeros(scene_cnt,2), 'MIAE': torch.zeros(0), 'MOAE': torch.zeros(0)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']
        pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt= compute_metrics_single_scene(pre_dict, gt_dict,intervals)
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])
        metrics['MIAE'] =  torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:, 0] - inflow_cnt[:, 1])])
        metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])
    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:, 0] * (metrics['WRAE'][:, 1] / (metrics['WRAE'][:, 1].sum() + 1e-10))) * 100
    MIAE = torch.mean(metrics['MIAE'] )
    MOAE = torch.mean(metrics['MOAE'])
    return MAE,MSE, WRAE,MIAE,MOAE,metrics['MAE']

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='HT21')
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--output_dir', type=str, default='saved_den_ht21')
    parser.add_argument('--task', type=str, default='DEN', choices=['DEN', 'LAB']) # LAB for colorization or DEN for density regression
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    setup_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.VAL_VIS_PATH = os.path.join(args.output_dir, 'val_vis')
    if args.type_dataset == 'SENSE':
        args.VAL_INTERVALS = 8
    else:
        args.VAL_INTERVALS = 50
    if args.type_dataset == 'HT21':
        args.VAL_FREQ = 1
        args.VAL_DENSE_START = 2
    else:
        args.VAL_FREQ = 1
        args.VAL_DENSE_START = 0
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    cc_trainer = Trainer(datasetting.cfg_data, args)
    cc_trainer.forward()