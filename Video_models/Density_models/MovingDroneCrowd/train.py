import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from misc import tools
from torch import optim
from misc.utils import adjust_learning_rate, AverageMeter, reduce_dict, save_visual_results, save_test_visual, compute_metrics_all_scenes, update_model, print_NWPU_summary_det
from copy import deepcopy
import torch.nn.functional as F
from torch.nn import SyncBatchNorm
from model.VIC import Video_Counter
from misc.tools import is_main_process
import timm.optim.optim_factory as optim_factory
import argparse
import os
import numpy as np
import torch
import datasets
from config import cfg
from importlib import import_module

def setup_seed(seed):
    tools.set_randomseed(seed + tools.get_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

class Trainer():
    def __init__(self, cfg_data, args):
        self.cfg_data = cfg_data
        self.args = args
        # model
        self.model = self.model_without_ddp = Video_Counter(cfg, cfg_data)
        self.model.cuda()
        self.val_frame_intervals = cfg_data.VAL_FRAME_INTERVALS
        if cfg.distributed:
            sync_model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(sync_model, device_ids=[cfg.gpu], find_unused_parameters=False)
            self.model_without_ddp = self.model.module
        # train and val loader
        self.train_loader, self.sampler_train, self.val_loader, self.restore_transform = datasets.loading_data(args.type_dataset, self.val_frame_intervals, cfg.distributed, is_main_process())
        # optimizer
        param_groups = optim_factory.add_weight_decay(self.model_without_ddp, cfg.WEIGHT_DECAY)
        self.optimizer = optim.Adam(param_groups, lr=cfg.LR_Base)
        self.i_tb = 0
        self.epoch = 1
        self.num_iters = cfg.MAX_EPOCH * int(len(self.train_loader))
        self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE': 1e20, 'WRAE': 1e20, 'MIAE': 1e20, 'MOAE': 1e20, 'share_mae': 1e20, 'share_mse': 1e20}
        if args.resume:
            latest_state = torch.load(args.resume_dir)
            self.model.load_state_dict(latest_state['net'], strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch']
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            print('Load ckpt from:', args.resume_dir)
        if cfg.PRE_TRAIN_COUNTER:
            counting_pre_train = torch.load(cfg.PRE_TRAIN_COUNTER)
            model_dict = self.model.state_dict()
            new_dict = {}
            for k, v in counting_pre_train.items():
                if 'Extractor' in k or 'global_decoder' in k:
                    if 'module' in k:
                        if cfg.distributed:
                            new_dict[k] = v
                        else:
                            new_dict[k[7:]] = v 
                    else:
                        if cfg.distributed:
                            new_dict['module.' + k] = v
                        else:
                            new_dict[k] = v 
            model_dict.update(new_dict)
            self.model.load_state_dict(model_dict, strict=True)
            print('Load ckpt of counter from:', cfg.PRE_TRAIN_COUNTER)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH + 1):
            self.epoch = epoch
            if cfg.distributed:
                self.sampler_train.set_epoch(epoch)
            self.train()
            if epoch % cfg.VAL_INTERVAL == 0 and epoch >= cfg.START_VAL:
                if is_main_process():
                    self.validate()
                if cfg.distributed:
                    torch.distributed.barrier()

    def train(self):
        self.model.train()
        lr = adjust_learning_rate(self.optimizer, cfg.LR_Base, self.num_iters, self.i_tb)
        batch_loss = {}
        for i, data in enumerate(self.train_loader, 0):
            self.i_tb += 1
            img, target = data
            for i in range(len(target)):
                for key, data in target[i].items():
                    if torch.is_tensor(data):
                        target[i][key]=data.cuda()
            img = img.cuda() # [2, 3, 768, 1024]
            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, loss_dict = self.model(img, target) # [2, 1, 768, 1024]
            pre_global_cnt = pre_global_den.sum()
            gt_global_cnt = gt_global_den.sum()
            all_loss = 0
            for v in loss_dict.values():
                all_loss += v
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()
            loss_dict_reduced = reduce_dict(loss_dict)
            for k, v in loss_dict_reduced.items():
                if not k in batch_loss:
                    batch_loss[k] = AverageMeter()
                batch_loss[k].update(v.item())
            if self.i_tb % cfg.PRINT_FREQ == 0:
                if is_main_process():
                    loss_str = ''.join([f"[loss_{key} {value.avg:.4f}]" for key, value in batch_loss.items()])
                    print('Epoch: {}, Iter: [{}/{}], Loss: {}, Lr: {:.4f}'.format(self.epoch, self.i_tb, len(self.train_loader), loss_str, lr))
                    print('GT: {:.2f}, Pred: {:.2f}, Max pred: {:.2f}, Max gt: {:.2f}'.format(gt_global_cnt.item(), pre_global_cnt.item(),
                          pre_global_den.max().item() * self.cfg_data.DEN_FACTOR, gt_global_den.max().item() * self.cfg_data.DEN_FACTOR))
            if self.i_tb % 100 == 0:
                save_visual_results([img, gt_global_den, pre_global_den, gt_share_den, pre_share_den, gt_in_out_den, pre_in_out_den], self.restore_transform,
                                    os.path.join(self.args.output_dir, "training_visual"),  self.i_tb, int(os.environ['RANK']) if 'RANK' in os.environ else 0)

    def validate(self):
        self.model.eval()
        global_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
        scenes_pred_dict = []
        scenes_gt_dict = []
        for scene_id, (scene_name, sub_valset) in enumerate(self.val_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + self.val_frame_intervals
            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict  = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            visual_maps = []
            imgs = []
            for vi, data in enumerate(gen_tqdm, 0):
                if vi % self.val_frame_intervals == 0 or vi == len(sub_valset)-1:
                    frame_signal = 'match'
                else: 
                    frame_signal = 'skip'
                if frame_signal == 'match':
                    img, label = data
                    for i in range(len(label)):
                        for key, data in label[i].items():
                            if torch.is_tensor(data):
                                label[i][key] = data.cuda()
                    img = img.cuda() # [2, 3, 1080, 1920]
                    with torch.no_grad():
                        b, c, h, w = img.shape 
                        if h % 32 != 0:
                            pad_h = 32 - h % 32
                        else: pad_h = 0
                        if w % 32 != 0:
                            pad_w = 32 - w % 32
                        else:
                            pad_w = 0
                        pad_dims = (0, pad_w, 0, pad_h)
                        img = F.pad(img, pad_dims, "constant") # [2, 3, 1080, 1920]
                        h, w = img.size(2), img.size(3)
                        place_holder_img = torch.zeros((1, h, w)).cuda()
                        if cfg.distributed:
                            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, loss_dict = self.model.module(img, label)
                        else:
                            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, loss_dict = self.model(img, label) # [2, 1, 1088, 1920]
                        pre_in_out_den[pre_in_out_den < 0] = 0
                        gt_global_cnt, pre_global_cnt = gt_global_den[0].sum().item(),  pre_global_den[0].sum().item()
                        s_mae = abs(gt_global_cnt - pre_global_cnt)
                        s_mse = ((gt_global_cnt - pre_global_cnt) * (gt_global_cnt - pre_global_cnt))
                        global_cnt_errors['mae'].update(s_mae)
                        global_cnt_errors['mse'].update(s_mse)
                        if vi == 0:
                            pred_dict['first_frame'] = pre_global_cnt
                            gt_dict['first_frame'] = gt_global_cnt
                        pred_dict['inflow'].append(pre_in_out_den[1].sum().item())
                        pred_dict['outflow'].append(pre_in_out_den[0].sum().item())
                        gt_dict['inflow'].append(gt_in_out_den[1].sum().item())
                        gt_dict['outflow'].append(gt_in_out_den[0].sum().item())
                        if vi % self.val_frame_intervals == 0:
                            img0 = img[0]
                            gt_global_den0 = gt_global_den[0]
                            pre_global_den0 = pre_global_den[0]
                            if vi == 0:
                                gt_share_den_before = deepcopy(place_holder_img)
                                pre_share_den_before = deepcopy(place_holder_img)
                                gt_in_den = deepcopy(place_holder_img)
                                pre_in_den = deepcopy(place_holder_img)
                            else:
                                gt_share_den_before = previous_gt_share_den[1]
                                pre_share_den_before = previous_pre_share_den[1]
                                gt_in_den = previous_gt_in_out_den[1]
                                pre_in_den = previous_pre_in_out_den[1]
                            gt_share_den_next = gt_share_den[0]
                            pre_share_den_next = pre_share_den[0]
                            gt_out_den = gt_in_out_den[0]
                            pre_out_den = pre_in_out_den[0]
                            visual_map = torch.stack([gt_global_den0, pre_global_den0, gt_share_den_before, pre_share_den_before, gt_in_den, pre_in_den, gt_share_den_next,
                                                      pre_share_den_next, gt_out_den, pre_out_den], dim=0) # [10, 1, 1088, 1920]
                            visual_maps.append(visual_map)
                            imgs.append(img0)
                            previous_gt_share_den = gt_share_den
                            previous_pre_share_den = pre_share_den
                            previous_gt_in_out_den = gt_in_out_den
                            previous_pre_in_out_den = pre_in_out_den
                            if (vi + self.val_frame_intervals) > (len(sub_valset) - 1):
                                visual_map = torch.stack([gt_global_den[1], pre_global_den[1], gt_share_den[1], pre_share_den[1], gt_in_out_den[1], pre_in_out_den[1],
                                                          deepcopy(place_holder_img), deepcopy(place_holder_img), deepcopy(place_holder_img), deepcopy(place_holder_img)], dim=0) # [10, 1, 1088, 1920]
                                visual_maps.append(visual_map)
                                imgs.append(img[1])
            visual_maps = torch.stack(visual_maps, dim=0) # [8, 10, 1, 1088, 1920]
            save_test_visual(visual_maps, imgs, scene_name, self.restore_transform, os.path.join(self.args.output_dir, "val_visual", scene_name), self.epoch,
                             int(os.environ['RANK']) if cfg.distributed else 0)
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)
        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, 1)
        print('MAE: {:.2f}, MSE: {:.2f}, WRAE: {:.2f}, WIAE: {:.2f}, WOAE: {:.2f}'.format(MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
        print('Pre vs GT:', cnt_result)
        mae = global_cnt_errors['mae'].avg
        mse = np.sqrt(global_cnt_errors['mse'].avg)
        self.train_record = update_model(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE}, self.args)
        print_NWPU_summary_det(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE})
        torch.cuda.empty_cache()
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='MovingDroneCrowd')
    parser.add_argument('--seed', type=int, default=3035)
    parser.add_argument('--output_dir', type=str, default='saved_drone')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tools.init_distributed_mode(cfg)
    setup_seed(args.seed)
    print('Training dataset:', args.type_dataset)
    datasetting = import_module(f'datasets.setting.{args.type_dataset}')
    cc_trainer = Trainer(datasetting.cfg_data, args)
    cc_trainer.forward()
