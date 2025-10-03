from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model.locator import Crowd_locator
from model.PBM import BinarizedModule
from config import cfg
from misc.utils import adjust_learning_rate, vis_results, AverageMeter, AverageCategoryMeter, update_model, print_NWPU_summary
import numpy as np
import torch
import os
import datasets
import cv2
from tqdm import tqdm
from misc.compute_metric import eval_metrics
from misc.EMA import EMA

class Trainer():
    def __init__(self, cfg_data, args):
        self.cfg_data = cfg_data
        self.args = args
        # train and test loader
        self.src_train_loader, _, _ = datasets.loading_data(args.src_dataset, args)
        _, _, self.tra_restore_transform = datasets.loading_data(args.target_dataset, args)
        _, self.SHHB_loader, _ = datasets.loading_data(args.target_dataset, args)
        # model
        self.net = Crowd_locator(cfg.NET, cfg.GPU_ID)
        self.pseudo_head = BinarizedModule(768).cuda()
        self.worst_head = BinarizedModule(768).cuda()
        self.ema = EMA(self.net, 0.99)
        self.ema.register()
        # optimizer
        if cfg.OPT == 'Adam':
            self.optimizer = optim.Adam([{'params': self.net.Extractor.parameters(), 'lr': cfg.LR_BASE_NET, 'weight_decay': 1e-5},
                                         {'params': self.net.Binar.parameters(), 'lr': cfg.LR_BM_NET}, {'params': self.worst_head.parameters(), 'lr': cfg.LR_BM_NET}])
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        self.train_record = {'best_F1': 0, 'best_Pre': 0,'best_Rec': 0, 'best_mae': 1e20, 'best_mse': 1e20, 'best_nae': 1e20, 'best_model_name': ''}
        self.epoch = 0
        self.i_tb = 0
        self.num_iters = cfg.MAX_EPOCH * int(len(self.src_train_loader))
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.num_iters = latest_state['num_iters']
            self.train_record = latest_state['train_record']
            print('Load ckpt from:', cfg.RESUME_PATH)

    def forward(self):
        self.validate(self.SHHB_loader)
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.source_loader = enumerate(self.src_train_loader)
            self.epoch = epoch
            self.MultiHeadDebiased_train()
            if epoch % 1 == 0:
                self.validate(self.SHHB_loader)

    def dice_loss(self, target, predictive, ep=1e-8):
        intersection = 2 * torch.sum(predictive * target) + ep
        union = torch.sum(predictive) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return loss

    def MultiHeadDebiased_train(self):
        self.net.train()
        for i, data in enumerate(self.src_train_loader, 0):
            tra_data = self.source_loader.__next__()
            self.i_tb += 1
            img, strong_img, gt_map = data
            _, eff_data = tra_data
            tra_img, tra_strong_img, tra_gt = eff_data
            tra_img = Variable(tra_img).cuda()
            tra_gt = Variable(tra_gt).cuda()
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            batch_size = img.size(0)
            mix_img, mix_gt = torch.cat((img, tra_img)), torch.cat((gt_map, tra_gt))
            while True:
                T, P, B, Feat = self.net(mix_img, None, 'pseudo')
                threshold_matrix, pre_map, binar_map = T[: batch_size], P[: batch_size], B[: batch_size]
                head_map_loss, binar_map_loss = F.mse_loss(pre_map, gt_map), torch.abs(binar_map - gt_map).mean()
                sup_loss = head_map_loss + binar_map_loss
                tra_threshold_matrix, tra_pre_map, tra_binar_map = T[batch_size:], P[batch_size:], B[batch_size:]
                tra_feature = Feat[batch_size:]
                break
            while True:
                _, psuedo_binar_map = self.pseudo_head(tra_feature, tra_pre_map)
                L2_loss = F.mse_loss(psuedo_binar_map, tra_binar_map.detach())
                DEbias_L1_loss = torch.abs(psuedo_binar_map - tra_binar_map.detach()).mean()
                debiased_loss = L2_loss + DEbias_L1_loss
                break
            while True:
                with torch.no_grad():
                    self.ema.apply_shadow()
                    psu_threshold_matrix, psu_pre_map, psu_binar_map = self.net(tra_img, mask_gt = None, mode='val')
                    self.ema.restore()
                    psu_pre_map, psu_binar_map = psu_pre_map.detach(), psu_binar_map.detach()
                break
            consis_loss = F.mse_loss(tra_pre_map, psu_pre_map) + torch.abs(tra_binar_map - psu_binar_map).mean()
            all_loss = consis_loss + sup_loss + debiased_loss
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()
            lr1, lr2 = adjust_learning_rate(self.optimizer, cfg.LR_BASE_NET, cfg.LR_BM_NET, self.num_iters, self.i_tb)
            if (i + 1) % cfg.PRINT_FREQ == 0:
                print('Epoch: {}, Iter: [{}/{}, Consis loss: {:.2f}, Sup loss: {:.2f}, L2 loss: {:.2f}, DEbias loss: {:.2f}, Threshold: [{:.2f}|{:.2f}], Lr1: {:.2f}, Lr2: {:.2f}'.
                      format(self.epoch + 1, i + 1, len(self.src_train_loader), consis_loss.item(), sup_loss.item(), L2_loss.item(), DEbias_L1_loss.item(),
                             tra_threshold_matrix.mean().item(), psu_threshold_matrix.mean().item(), lr1, lr2))
            if  i % 100 == 0:
                box_pre, boxes = self.get_boxInfo_from_Binar_map(binar_map[0].detach().cpu().numpy())
                if not os.path.exists(self.args.vis_dir):
                    os.makedirs(self.args.vis_dir)
                vis_results(self.args.vis_dir, 0, self.tra_restore_transform, tra_img, tra_pre_map[0].detach().cpu().numpy(), tra_gt[0].detach().cpu().numpy(),
                            tra_binar_map.detach().cpu().numpy(), tra_threshold_matrix.detach().cpu().numpy(),boxes)

    def get_boxInfo_from_Binar_map(self, Binar_numpy, min_area=3):
        Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
        assert Binar_numpy.ndim == 2
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)
        boxes = stats[1:, :]
        points = centroids[1:, :]
        index = (boxes[:, 4] >= min_area)
        boxes = boxes[index]
        points = points[index]
        pre_data = {'num': len(points), 'points': points}
        return pre_data, boxes

    def validate(self, loader):
        self.net.eval()
        num_classes = 6
        losses = AverageMeter()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
        metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes), 'fn_c': AverageCategoryMeter(num_classes)}
        metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes), 'fn_c': AverageCategoryMeter(num_classes)}
        gen_tqdm = tqdm(loader)
        recall_c_list = [[] for _ in range(num_classes)]
        for vi, data in enumerate(gen_tqdm, 0):
            img,dot_map, gt_data = data
            slice_h, slice_w = 512, 512
            with torch.no_grad():
                img = Variable(img).cuda()
                dot_map = Variable(dot_map).cuda()
                crop_imgs, crop_gt, crop_masks = [], [], []
                b, c, h, w = img.shape
                if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                    [pred_threshold, pred_map, __] = [i.cpu() for i in self.net(img, mask_gt=None, mode = 'val')]
                else:
                    if h % 16 != 0:
                        pad_dims = (0, 0, 0, 16 - h % 16)
                        h = (h // 16 + 1) * 16
                        img = F.pad(img, pad_dims, "constant")
                        dot_map = F.pad(dot_map, pad_dims, "constant")
                    if w % 16 !=0:
                        pad_dims = (0, 16 - w % 16, 0, 0)
                        w =  (w // 16 + 1) * 16
                        img = F.pad(img, pad_dims, "constant")
                        dot_map = F.pad(dot_map, pad_dims, "constant")
                    assert img.size()[2:] == dot_map.size()[2:]
                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                            crop_gt.append(dot_map[:, :, h_start:h_end, w_start:w_end])
                            mask = torch.zeros_like(dot_map).cpu()
                            mask[:, :,h_start:h_end, w_start:w_end].fill_(1.0)
                            crop_masks.append(mask)
                    crop_imgs, crop_gt, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_gt, crop_masks))
                    crop_preds, crop_thresholds = [], []
                    nz, period = crop_imgs.size(0), 12
                    for i in range(0, nz, period):
                        [crop_threshold, crop_pred, __] = [i.cpu() for i in self.net(crop_imgs[i:min(nz, i + period)], mask_gt=None, mode='val')]
                        crop_preds.append(crop_pred)
                        crop_thresholds.append(crop_threshold)
                    crop_preds = torch.cat(crop_preds, dim=0)
                    crop_thresholds = torch.cat(crop_thresholds, dim=0)
                    idx = 0
                    pred_map = torch.zeros_like(dot_map).cpu().float()
                    pred_threshold = torch.zeros_like(dot_map).cpu().float()
                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            pred_map[:, :, h_start:h_end, w_start:w_end]  += crop_preds[idx]
                            pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                            idx += 1
                    mask = crop_masks.sum(dim=0)
                    pred_map = (pred_map / mask)
                    pred_threshold = (pred_threshold/mask)
                a = torch.ones_like(pred_map)
                b = torch.zeros_like(pred_map)
                binar_map = torch.where(pred_map >= pred_threshold, a, b)
                dot_map = dot_map.cpu()
                loss = F.mse_loss(pred_map, dot_map)
                losses.update(loss.item())
                binar_map = binar_map.numpy()
                pred_data,boxes = self.get_boxInfo_from_Binar_map(binar_map)
                tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l = eval_metrics(num_classes,pred_data,gt_data)
                metrics_s['tp'].update(tp_s)
                metrics_s['fp'].update(fp_s)
                metrics_s['fn'].update(fn_s)
                metrics_s['tp_c'].update(tp_c_s)
                metrics_s['fn_c'].update(fn_c_s)
                metrics_l['tp'].update(tp_l)
                metrics_l['fp'].update(fp_l)
                metrics_l['fn'].update(fn_l)
                metrics_l['tp_c'].update(tp_c_l)
                metrics_l['fn_c'].update(fn_c_l)
                for c in range(len(tp_c_l)):
                    recall_c_list[c].append(tp_c_l[c] / (tp_c_l[c]+fn_c_l[c] + 1e-5))
                gt_count, pred_cnt = gt_data['num'].numpy().astype(float), pred_data['num']
                s_mae = abs(gt_count - pred_cnt)
                s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                cnt_errors['mae'].update(s_mae)
                cnt_errors['mse'].update(s_mse)
                if gt_count != 0:
                    s_nae = (abs(gt_count - pred_cnt) / gt_count)
                    cnt_errors['nae'].update(s_nae)
                if vi == 0:
                    vis_results(self.args.vis_dir, self.epoch, self.tra_restore_transform, img, pred_map.numpy(), dot_map.numpy(),binar_map, pred_threshold.numpy(),boxes)
        ap_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fp'].sum + 1e-20)
        ar_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fn'].sum + 1e-20)
        f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s + 1e-20)
        ar_c_s = metrics_s['tp_c'].sum / (metrics_s['tp_c'].sum + metrics_s['fn_c'].sum + 1e-20)
        ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
        ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
        f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l + 1e-20)
        ar_c_l = metrics_l['tp_c'].sum / (metrics_l['tp_c'].sum + metrics_l['fn_c'].sum + 1e-20)
        loss = losses.avg
        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg
        print('Val loss: {:.2f}, F1: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, MAE: {:.2f}, MSE: {:.2f}, NAE: {:.2f}'.format(loss, f1m_l, ap_l, ar_l, mae[0], mse[0], nae[0]))
        self.train_record = update_model(self, [f1m_l, ap_l, ar_l, mae, mse, nae, loss], self.args)
        print_NWPU_summary(self,[f1m_l, ap_l, ar_l,mae, mse, nae, loss])
        # content = f"{self.epoch}\t"
        # for c in range(len(tp_c_l)):
        #     recall_c_list[c] = np.mean(recall_c_list[c])
        #     content += "{:.2f}".format(recall_c_list[c])
        #     content += '\t'
        # content += '\n'
        # with open('Class Recall.txt', 'a') as f:
        #     f.write(content)