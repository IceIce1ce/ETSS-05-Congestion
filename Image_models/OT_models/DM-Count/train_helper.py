import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh
from models import vgg19
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        self.save_dir = os.path.join('checkpoints', args.output_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        log_utils.print_config(vars(args), self.logger)
        downsample_ratio = 8
        if args.type_dataset.lower() == 'qnrf':
            self.datasets = {x: Crowd_qnrf(os.path.join(args.dataset_dir, x), args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.type_dataset.lower() == 'nwpu':
            self.datasets = {x: Crowd_nwpu(os.path.join(args.dataset_dir, x), args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.type_dataset.lower() == 'sha' or args.type_dataset.lower() == 'shb':
            self.datasets = {'train': Crowd_sh(os.path.join(args.dataset_dir, 'train_data'), args.crop_size, downsample_ratio, 'train'),
                             'val': Crowd_sh(os.path.join(args.dataset_dir, 'test_data'), args.crop_size, downsample_ratio, 'val')}
        else:
            print('This dataset does not exist')
            raise NotImplementedError
        # dataloader (1 is num_device)
        self.dataloaders = {x: DataLoader(self.datasets[x], collate_fn=(train_collate if x == 'train' else default_collate), batch_size=(args.batch_size if x == 'train' else 1),
                            shuffle=(True if x == 'train' else False), num_workers=args.num_workers * 1, pin_memory=(True if x == 'train' else False)) for x in ['train', 'val']}
        # model
        self.model = vgg19()
        self.model.cuda()
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.start_epoch = 0
        if args.resume:
            self.logger.info('Loading ckpt from:', args.resume)
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, map_location='cuda')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location='cuda'))
        else:
            self.logger.info('random initialization')
        # loss
        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, args.num_iter_ot, args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').cuda()
        self.mse = nn.MSELoss().cuda()
        self.mae = nn.L1Loss().cuda()
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs + 1):
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.epochs) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()
        for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders['train']):
            inputs = inputs.cuda() # [10, 3, 384, 384]
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.cuda() for p in points] # [7, 2] * len(10)
            gt_discrete = gt_discrete.cuda() # [10, 1, 48, 48]
            N = inputs.size(0) # 10
            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs) # [10, 1, 48, 48], [10, 1, 48, 48]
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)
                count_loss = self.mae(outputs.sum(1).sum(1).sum(1), torch.from_numpy(gd_count).float().cuda())
                epoch_count_loss.update(count_loss.item(), N)
                gd_count_tensor = torch.from_numpy(gd_count).float().cuda().unsqueeze(1).unsqueeze(2).unsqueeze(3) # [10, 1, 1, 1]
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6) # [10, 1, 48, 48]
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(1) * torch.from_numpy(gd_count).float().cuda()).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)
                loss = ot_loss + count_loss + tv_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy() # [10]
                pred_err = pred_count - gd_count # [10]
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)
        self.logger.info('Epoch: [{}, {}], Training loss: {:.4f}, OT Loss: {:.4e}, Wass Distance: {:.4f}, OT obj value: {:.4f}, Count Loss: {:.4f}, TV Loss: {:.4f}, MSE: {:.4f} MAE: {:.4f}, Cost {:.1f} sec'
                         .format(self.epoch + 1, self.args.epochs, epoch_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(), epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(), time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({'epoch': self.epoch, 'optimizer_state_dict': self.optimizer.state_dict(), 'model_state_dict': model_state_dic}, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.cuda() # [1, 3, 1264, 1920]
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, _ = self.model(inputs) # [1, 1, 158, 240]
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res) # [500]
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info('Epoch: [{}, {}], MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'.format(self.epoch + 1, self.args.epochs, mse, mae, time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("Saving best mse: {:.4f} mae {:.2f} at epoch {}".format(self.best_mse, self.best_mae, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1