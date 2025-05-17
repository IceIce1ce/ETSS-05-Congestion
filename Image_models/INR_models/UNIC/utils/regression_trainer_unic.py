from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.aspd_spatial_uq1 import New_bay_Net
from datasets.crowd_unic import Crowd
from losses.bay_loss_new import Bay_Loss
from losses.post_prob_duo import Post_Prob
import random
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') # fix main thread is not in main loop error

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_parameters_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    prior_prob = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    grid_c = torch.stack(transposed_batch[4], 0)
    gridnum_sam_c = transposed_batch[5]
    gd_count = transposed_batch[6]
    return images, points, prior_prob, st_sizes, grid_c, gridnum_sam_c, gd_count

class RegTrainer(Trainer):
    def setup(self):
        args = self.args
        self.downsample_ratio = args.downsample_ratio
        # dataloader
        self.datasets = {x: Crowd((os.path.join(args.input_dir, 'train_data/images') if x == 'train' else os.path.join(args.input_dir, 'test_data/images')),
                         args.crop_size, args.downsample_ratio, args.is_gray, x) for x in ['train', 'val']}
        g = torch.Generator()
        g.manual_seed(args.seed)
        # 1 is num gpus
        self.dataloaders = {x: DataLoader(self.datasets[x], collate_fn=(train_collate if x == 'train' else default_collate), batch_size=(args.batch_size if x == 'train' else 1),
                            shuffle=(True if x == 'train' else False), num_workers=args.num_workers * 1, pin_memory=(True if x == 'train' else False),
                            worker_init_fn=(seed_worker if x == 'train' else None), generator=(g if x == 'train' else None)) for x in ['train', 'val']}
        # model
        self.model = New_bay_Net(args.crop_size)
        self.model.cuda()
        total_num, trainable_num = get_parameters_number(self.model)
        print('Total params:', total_num)
        print('Total trainable params:', trainable_num)
        # optimizer
        c_params = list(map(id, self.model.cc_decoder.last2.parameters()))
        b_params = filter(lambda p: id(p) not in c_params, self.model.parameters())
        self.optimizer1 = optim.Adam(b_params, lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer2 = optim.Adam(self.model.cc_decoder.last2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = lr_scheduler.StepLR(self.optimizer1, step_size = 3000, gamma = 1)
        self.start_epoch = 0
        self.epoch = self.start_epoch
        # resume training
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, map_location='cuda')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer1.load_state_dict(checkpoint['optimizer_state_dict1'])
                self.optimizer2.load_state_dict(checkpoint['optimizer_state_dict2'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location='cuda'))
            elif suf == 'pt':
                self.model.load_state_dict(torch.load(args.resume, map_location='cuda'))
        # loss
        self.post_prob = Post_Prob(args.sigma, args.crop_size, args.downsample_ratio, args.background_ratio, args.use_background)
        self.criterion = Bay_Loss(args.use_background)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.epochs - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch(self.epoch)
            self.scheduler.step()
            # if epoch % args.val_epoch == 0 and epoch >= args.val_start:
            if epoch % 1 == 0:
                self.val_epoch()

    def train_eopch(self, epoch):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        self.model.train()
        for step, (inputs, points, prior_prob, st_sizes, grid_c, gridnum_sam_c, gd_count) in enumerate(self.dataloaders['train']):
            inputs = inputs.cuda() # [16, 3, 256, 256]
            points = [p.cuda() for p in points] # [3138, 2] * len(16)
            prior_prob = [t.cuda() for t in prior_prob] # [3140, 1024] * len(16)
            st_sizes = st_sizes.cuda() # [16]
            grid_c = grid_c.cuda() # [16, 1024, 2]
            gridnum_sam_c = [tt.cuda() for tt in gridnum_sam_c] # [1024] * len(16)
            with torch.set_grad_enabled(True):
                outputs, out_sigma = self.model(inputs, grid_c, 'train') # [16, 1024]
                prob_list = self.post_prob(points, st_sizes, gridnum_sam_c) # [3139, 1024] * len(16)
                loss1 = self.criterion(prob_list, prior_prob, outputs)
                loss_KL = self.model.kl_div
                outputs = outputs / 10
                if False:
                    loss = 1.0 * loss1 + 0.01 * loss_KL
                    if epoch < 500:
                        if epoch % 5 == 1:
                            self.optimizer1.zero_grad()
                            self.optimizer2.zero_grad()
                            loss.backward()
                            self.optimizer2.step()
                        else:
                            self.optimizer1.zero_grad()
                            self.optimizer2.zero_grad()
                            loss.backward()
                            self.optimizer1.step()
                    else:
                        self.optimizer1.zero_grad()
                        self.optimizer2.zero_grad()
                        loss.backward()
                        self.optimizer1.step()
                        self.optimizer2.step()
                else:
                  loss = 1.0 * loss1 # + 0.1 * loss_KL
                  self.optimizer1.zero_grad()
                  self.optimizer2.zero_grad()
                  loss.backward()
                  self.optimizer1.step()
                  self.optimizer2.step()
                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
        logging.info('Training: Epoch: {}, Loss: {:.4f}, MSE: {:.4f}, MAE: {:.4f}'.format(self.epoch + 1, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg()))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({'epoch': self.epoch, 'optimizer_state_dict1': self.optimizer1.state_dict(), 'optimizer_state_dict2': self.optimizer2.state_dict(), 'model_state_dict': model_state_dic}, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        self.model.eval()
        epoch_res = []
        sr_results = []
        c_results = []
        for inputs, count, name, cor_C in self.dataloaders['val']:
            inputs = inputs.cuda() # [1, 3, 256, 256]
            cor_C = cor_C.cuda() # [1, 1024, 2]
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, out_sigma = self.model(inputs, cor_C, 'test') # [1024]
                outputs = outputs / 10
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
                c_results.append(outputs.data.cpu().numpy())
        epoch_res = np.array(epoch_res) # [182]
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Testing: Epoch: {}, MSE: {:.4f}, MAE: {:.4f}'.format(self.epoch, mse, mae))
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("Save best MSE: {:.4f}, MAE: {:.2f} at epoch: {}".format(self.best_mse, self.best_mae, self.epoch))
            torch.save(self.model, os.path.join(self.save_dir, 'best_model.pt'))
            sr_results = np.array(sr_results) # [0]
            c_results = np.array(c_results) # [182, 1024]
            np.save('srr_image.npy', sr_results)
            np.save('cc_image.npy', c_results)
        logging.info("Best MSE: {:.4f}, MAE: {:.4f} at epoch {}".format(self.best_mse, self.best_mae, self.epoch))
        if mae < 7.5:
            torch.save(self.model, os.path.join(self.save_dir, 'recent_model.pt'))
        if mae < 7.2:
            torch.save(self.model, os.path.join(self.save_dir, 'fine_model.pt'))
        c_results = np.array(c_results) # [182, 1024]
        if self.epoch % 10 == 0:
            fig = plt.figure(figsize=(10, 7))
            rows = 1
            columns = 4
            for i in range(4):
                c_example = np.reshape(c_results[i], [32, 32])
                fig.add_subplot(rows, columns, i + 1)
                plt.imshow(c_example)
            plt.savefig(os.path.join(self.save_dir, str(self.epoch) + '_test_fig.png'))
            plt.close()