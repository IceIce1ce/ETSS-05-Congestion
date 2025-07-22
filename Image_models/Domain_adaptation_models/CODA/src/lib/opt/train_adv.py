import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from src.lib.opt.lr_policy import update_lr
from src.lib.dataset.shanghaitech import HeadCountDataset, IsColor, RandomFlip, PreferredSize, ToTensor, Normalize, NineCrop, Multiscale
from torchvision import transforms
from torch.utils.data import DataLoader
from src.lib.network.cn import vgg16
from src.lib.network.discriminator import Discriminator
import os
import sys

class TrainModel(object):
    def __init__(self, data_path, target_data_path, batchsize, lr, epoch, snap_shot, server_root_path, start_epoch=0, steps=[], decay_rate=0.1, branch=vgg16, pre_trained=None, resize=896, test_size={'train': 10, 'test': 100}):
        self.test_size = test_size
        self.use_multiscale = True
        self.log_path = os.path.join(server_root_path, 'log', branch.name + '.log')
        if not os.path.exists(os.path.join(server_root_path, 'log')):
            os.makedirs(os.path.join(server_root_path, 'log'))
        self.lr_D = 1e-3
        self.save_path = server_root_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # train and test loader
        train_dataset = HeadCountDataset(epoch,'train_adv', os.path.join(data_path, 'train_data.txt'), use_pers=False, use_attention=False, transform=transforms.Compose([IsColor(True), NineCrop(), RandomFlip(),PreferredSize(resize), ToTensor(use_att=False), Normalize()]))
        target_dataset = HeadCountDataset(epoch,'train_adv',os.path.join(target_data_path, 'train_data.txt'), use_pers=False, use_attention=False, transform=transforms.Compose([IsColor(True), NineCrop(), RandomFlip(), Multiscale(cropscale=[0.5, 0.75]),
                                          PreferredSize(resize, use_multiscale=self.use_multiscale), ToTensor(use_att=False, use_multiscale=self.use_multiscale), Normalize(use_multiscale=self.use_multiscale)]))
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=32)
        target_dataloader = DataLoader(target_dataset, batch_size=batchsize, shuffle=True, num_workers=32)
        val_dataset = HeadCountDataset(epoch,'test', os.path.join(target_data_path, 'test_data.txt'), use_pers=False, use_attention=False, transform=transforms.Compose([IsColor(True), PreferredSize(1024), ToTensor(use_att=False), Normalize()]))
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=32)
        self.dataloader = {'train': train_dataloader, 'target': target_dataloader, 'val': val_dataloader}
        # model
        self.model = branch()
        self.model.create_architecture()
        self.start_epoch = start_epoch
        self.model_D = Discriminator(1)
        # optimzier
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=self.lr_D, betas=(0.9, 0.99))
        if pre_trained['density'] != '':
            self.loadModel(self.model, pre_trained['density'])
        if pre_trained['discriminator'] != '':
            self.loadModel(self.model_D, pre_trained['discriminator'])
        # loss
        self.criterion_dens = nn.MSELoss(size_average=False)
        self.criterion_disc = nn.BCEWithLogitsLoss()
        self.criterion_rank = nn.MarginRankingLoss()
        self.num_iter = epoch
        self.snap_shot = snap_shot
        self.steps = steps
        self.decay_rate = decay_rate
        self.power = 0.9
        self.source_label = 0
        self.target_label = 1
        self.lambda_adv = 0.001

    def lr_poly(self,base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def adjust_learning_rate_D(self,optimizer, i_iter):
        lr = self.lr_poly(self.lr_D, i_iter, self.num_iter, self.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def loadModel(self,model, model_path=None):
        if model_path != '':
            pretrained_dict = torch.load(model_path)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('Load ckpt from: {}'.format(model_path))

    def saveModel(self,model, save_path,epoch):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = save_path + '/{}.model'.format(str(epoch))
        torch.save(model.state_dict(), model_path)

    def train_model(self):
        trainloader_iter = enumerate(self.dataloader['train'])
        targetloader_iter = enumerate(self.dataloader['target'])
        best_mae = sys.maxsize
        loss_dens_value = 0.0
        loss_adv_value = 0.0
        loss_D_value = 0.0
        running_mse = 0.0
        running_mae = 0.0
        totalnum = 0
        iter_count = 0
        for epoch in range(self.start_epoch,self.num_iter+1):
            iter_count = iter_count + 1
            self.optimizer = update_lr(self.optimizer, epoch, self.steps, self.decay_rate)
            self.model.train(True)
            self.model_D.train(True)
            self.adjust_learning_rate_D(self.optimizer_D, epoch)
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()
            for param in self.model_D.parameters():
                param.requires_grad = False
            _, data = next(trainloader_iter)
            image = data['image'] # [16., 3, 512, 512]
            Dmap = data['densityMap'] # [16, 512, 512]
            image, Dmap = Variable(image.cuda(), requires_grad=False), Variable(Dmap.cuda(), requires_grad=False)
            self.model = self.model.cuda()
            self.model_D = self.model_D.cuda()
            predDmap = self.model(image) # [16, 1, 512, 512]
            loss = self.criterion_dens(torch.squeeze(predDmap, 1), Dmap)
            outputs_np = predDmap.data.cpu().numpy()
            Dmap_np = Dmap.data.cpu().numpy()
            iter_size = outputs_np.shape[0]
            totalnum += iter_size
            loss.backward()
            loss_dens_value += loss.data.item() / iter_size
            _, data_t = next(targetloader_iter)
            image_t = data_t['image'] # [16, 3, 512, 512]
            if self.use_multiscale:
                multi_img = data_t['scale_images'] # [16, 2, 3, 512, 512]
            image_t = Variable(image_t.cuda(), requires_grad=False)
            if self.use_multiscale:
                multi_img = Variable(multi_img.cuda(), requires_grad=False)
            self.model = self.model.cuda()
            self.model_D = self.model_D.cuda()
            predDmap_t = self.model(image_t) # [16, 1, 512, 512]
            D_out_t = self.model_D(predDmap_t) # [16, 1, 16, 16]
            loss = self.lambda_adv * self.criterion_disc(D_out_t, Variable(torch.FloatTensor(D_out_t.data.size()).fill_(self.source_label)).cuda())
            if self.use_multiscale:
                multi_imgs = torch.chunk(multi_img, multi_img.size()[1], dim=1)
                predDmap_t_subs = []
                D_out_t_subs = []
                for sub_img in multi_imgs:
                    sub_img = torch.squeeze(sub_img) # [16, 3, 512, 512]
                    predDmap_t_sub = self.model(sub_img) # [16, 1, 512, 512]
                    predDmap_t_subs.append(predDmap_t_sub)
                    D_out_t_sub = self.model_D(predDmap_t_sub) # [16, 1, 16, 16]
                    D_out_t_subs.append(D_out_t_sub)
                for i in range(len(multi_imgs)):
                    loss += self.lambda_adv*self.criterion_disc(D_out_t_subs[i], Variable(torch.FloatTensor(D_out_t_subs[i].data.size()).fill_(self.source_label)).cuda())
                    if i == 0:
                        pred_cnt = torch.sum(predDmap_t.reshape(predDmap_t.size(0), -1), dim=1)
                        pred_cnt_sub = torch.sum(predDmap_t_subs[i].reshape(predDmap_t_subs[i].size(0), -1), dim=1)
                        loss += self.lambda_adv*self.criterion_rank(pred_cnt_sub, pred_cnt, Variable(torch.Tensor(pred_cnt.data.size()).fill_(-1)).cuda())
                    else:
                        pred_cnt_sub1 = torch.sum(predDmap_t_subs[i - 1].reshape(predDmap_t_subs[i - 1].size(0), -1), dim=1)
                        pred_cnt_sub2 = torch.sum(predDmap_t_subs[i].reshape(predDmap_t_subs[i].size(0), -1), dim=1)
                        loss += self.lambda_adv * self.criterion_rank(pred_cnt_sub2, pred_cnt_sub1, Variable(torch.Tensor(pred_cnt_sub1.data.size()).fill_(-1)).cuda())
            loss.backward()
            loss_adv_value += loss / iter_size
            pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0], -1)), 1)
            gt_count = np.sum(Dmap_np.reshape((Dmap_np.shape[0], -1)), 1)
            running_mae += np.sum(np.abs(pre_dens - gt_count))
            running_mse += np.sum((pre_dens - gt_count) ** 2)
            for param in self.model_D.parameters():
                param.requires_grad = True
            predDmap = predDmap.detach()
            D_out = self.model_D(predDmap) # [16, 1, 16, 16]
            loss = self.criterion_disc(D_out,Variable(torch.FloatTensor(D_out.data.size()).fill_(self.source_label)).cuda())
            loss.backward()
            loss_D_value += loss.data.item() / iter_size
            predDmap_t = predDmap_t.detach()
            D_out_t = self.model_D(predDmap_t) # [16, 1, 16, 16]
            loss = self.criterion_disc(D_out_t, Variable(torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label)).cuda())
            loss.backward()
            loss_D_value += loss.data.item() / iter_size
            if self.use_multiscale:
                for i in range(len(predDmap_t_subs)):
                    predDmap_t_subs[i] = predDmap_t_subs[i].detach() # [16, 1, 512, 512]
                    D_out_t_sub = self.model_D(predDmap_t_subs[i]) # [16, 1, 16, 16]
                    loss = self.criterion_disc(D_out_t_sub,Variable(torch.FloatTensor(D_out_t_sub.data.size()).fill_(self.target_label)).cuda())
                    loss.backward()
                    loss_D_value += loss.data.item() /iter_size
            self.optimizer.step()
            self.optimizer_D.step()
            if epoch % self.test_size['train'] == 0:
                epoch_mae = running_mae / totalnum
                epoch_mse = np.sqrt(running_mse /totalnum)
                print('Train: Epoch: {}, MAE: {:.4f}, MSE: {:.4f}'.format(epoch, epoch_mae, epoch_mse))
                print('Train: Density loss: {:.4f}, Density adversarial loss: {:.4f}, Discriminator loss: {:.4f}'.format(loss_dens_value / iter_count, loss_adv_value / iter_count, loss_D_value / iter_count))
                f = open(self.log_path, 'a')
                f.write('Train: Epoch: {}, MAE: {:.4f}, MSE: {:.4f}\n'.format(epoch, epoch_mae, epoch_mse))
                f.close()
                running_mae = 0.0
                running_mse = 0.0
                totalnum= 0.0
                loss_dens_value = 0.0
                loss_adv_value = 0.0
                loss_D_value = 0.0
                iter_count = 0
            if epoch % self.test_size['test'] == 0:
                self.model.train(False)
                self.model.eval()
                self.model_D.train(False)
                self.model_D.eval()
                totalnum_test = 0.0
                running_mse_test = 0.0
                running_mae_test = 0.0
                running_loss_test = 0.0
                for idx,data in enumerate(self.dataloader['val']):
                    image = data['image'] # [16, 3, 1024, 1024]
                    densityMap = data['densityMap'] # [16, 1024, 1024]
                    image, densityMap = Variable(image.cuda(), requires_grad=False), Variable(densityMap.cuda(), requires_grad=False)
                    self.model = self.model.cuda()
                    predDensityMap = self.model(image) # [16, 1, 1024, 1024]
                    loss = self.criterion_dens(torch.squeeze(predDensityMap, 1), densityMap)
                    outputs_np = predDensityMap.data.cpu().numpy()
                    densityMap_np = densityMap.data.cpu().numpy()
                    pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0], -1)), 1)
                    gt_count = np.sum(densityMap_np.reshape((densityMap_np.shape[0], -1)), 1)
                    totalnum_test += outputs_np.shape[0]
                    running_mae_test += np.sum(np.abs(pre_dens - gt_count))
                    running_mse_test += np.sum((pre_dens - gt_count)**2)
                    running_loss_test += loss.data.item()
                epoch_loss = running_loss_test / totalnum_test
                epoch_mae = running_mae_test/totalnum_test
                epoch_mse = np.sqrt(running_mse_test/totalnum_test)
                f = open(self.log_path, 'a')
                print('Test: MAE: {:.4f}, MSE: {:.4f}'.format(epoch_mae, epoch_mse))
                f.write('Test: Epoch: {}, MAE: {:.4f}, MSE: {:.4f}\n'.format(epoch, epoch_mae, epoch_mse))
                f.close()
                if epoch % self.snap_shot == 0:
                    self.saveModel(self.model, os.path.join(self.save_path, 'Generator'), epoch)
                    self.saveModel(self.model_D, os.path.join(self.save_path, 'Discriminator'), epoch)
                    if best_mae > epoch_mae:
                        best_mae = epoch_mae
                        f = open(self.log_path, 'a')
                        f.write("+++++++++++++++++++++Best Density Epoch: {}++++++++++++++++++++++++\n".format(epoch))
                        f.write('Loss: {:.4f}, MAE: {:.4f}, MSE: {:.4f}\n'.format(epoch_loss, epoch_mae, epoch_mse))
                        f.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                        f.close()

    def run(self):
        self.train_model()