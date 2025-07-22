import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from src.lib.utils.image_opt import showTest
from src.lib.opt.lr_policy import update_lr
from src.lib.dataset.shanghaitech import HeadCountDataset, IsColor, RandomFlip, PreferredSize, ToTensor, Normalize, NineCrop
from torchvision import transforms
from torch.utils.data import DataLoader
from src.lib.network.cn import vgg16
import os
import sys

class TrainModel(object):
    def __init__(self, data_path, batchsize, lr, epoch, server_root_path, val_size, snap_shot, start_epoch=0, steps=[], decay_rate=0.1, branch=vgg16, pre_trained=None, resize=896):
        self.snap_shot = snap_shot
        self.log_path = os.path.join(server_root_path, 'log', branch.name + '.log')
        if not os.path.exists(os.path.join(server_root_path, 'log')):
            os.makedirs(os.path.join(server_root_path, 'log'))
        self.val_size = val_size
        self.save_path = server_root_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # train and test loader
        self.train_dataset = HeadCountDataset(data_file=os.path.join(data_path, 'train_data.txt'), use_pers=False, use_attention=False, transform=transforms.Compose([IsColor(True), NineCrop(), RandomFlip(), PreferredSize(resize), ToTensor(), Normalize()]), max_iter=1000, phase='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batchsize, shuffle=True, num_workers=32)
        self.val_dataset = HeadCountDataset(data_file=os.path.join(data_path,'test_data.txt'),use_pers=False,use_attention=False, transform=transforms.Compose([IsColor(True), PreferredSize(1024), ToTensor(use_att=False), Normalize()]), max_iter=1000, phase='val')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=4, shuffle=False, num_workers=32)
        self.dataloader = {'train': self.train_dataloader, 'val': self.val_dataloader}
        self.model = branch()
        self.model.create_architecture()
        self.start_epoch = start_epoch
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9)
        self.loadModel(pre_trained)
        self.criterion_dens = nn.MSELoss(size_average=False)
        self.num_epoch = epoch
        self.steps = steps
        self.decay_rate = decay_rate

    def loadModel(self, model_path=None):
        if model_path != '':
            pretrained_dict = torch.load(model_path)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            for k in pretrained_dict:
                print('key:',k)
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print('Load model:{}'.format(model_path))

    def saveModel(self, save_path, epoch):
        model_path = save_path + '/{}.model'.format(str(epoch))
        torch.save(self.model.state_dict(), model_path)

    def train_model(self, model, optimizer, num_epochs=25, val_size=50):
        best_mae = sys.maxsize
        for epoch in range(self.start_epoch, num_epochs + 1):
            for phase in ['train', 'val']:
                if phase == 'train':
                    optimizer = update_lr(optimizer, epoch, self.steps, self.decay_rate)
                    model.train(True)
                else:
                    if epoch % val_size != 0:
                        continue
                    model.train(False)
                    model.eval()
                running_loss = 0.0
                running_mse = 0.0
                running_mae = 0.0
                totalnum = 0
                for idx, data in enumerate(self.dataloader[phase]):
                    image = data['image'] # [16, 3, 512, 512]
                    densityMap = data['densityMap'] # [16, 512, 512]
                    image, densityMap = Variable(image.cuda(), requires_grad=False), Variable(densityMap.cuda(), requires_grad=False)
                    self.model = self.model.cuda()
                    optimizer.zero_grad()
                    predDensityMap = model(image) # [16, 1, 512, 512]
                    predDensityMap = torch.squeeze(predDensityMap) # [16, 512, 512]
                    loss = self.criterion_dens(predDensityMap, densityMap)
                    outputs_np = predDensityMap.data.cpu().numpy()
                    densityMap_np = densityMap.data.cpu().numpy()
                    pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0], -1)),1)
                    gt_count = np.sum(densityMap_np.reshape((densityMap_np.shape[0], -1)),1)
                    totalnum += outputs_np.shape[0]
                    running_mae += np.sum(np.abs(pre_dens - gt_count))
                    running_mse += np.sum((pre_dens - gt_count)**2)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        if idx == 20 and epoch % 100 == 0:
                            showTest(image.data.cpu(), predDensityMap.data.cpu(), densityMap.data.cpu(), idx, self.save_path)
                        if idx % 20 == 0 and phase == 'val':
                            f = open(self.log_path, 'a')
                            f.write('-------------------Density Count----------------------\n')
                            f.write("Epoch: %4d, Step: %4d, GT: %4.1f, Pred: %4.1f, Loss: %4.4f\n" % (epoch, idx, np.sum(gt_count), np.sum(pre_dens), loss.data.item()))
                            f.write('---------------------------------------------------\n')
                            f.close()
                    running_loss += loss.data.item()
                epoch_loss = running_loss / totalnum
                epoch_mae = running_mae / totalnum
                epoch_mse = np.sqrt(running_mse / totalnum)
                if epoch % self.snap_shot == 0:
                    self.saveModel(self.save_path,epoch)
                    if best_mae > epoch_mae and phase=='val':
                        best_mae = epoch_mae
                        f = open(self.log_path, 'a')
                        f.write("+++++++++++++++++++++Best Density Epoch: {}++++++++++++++++++++++++\n".format(epoch))
                        f.write('{}: Loss: {:.4f}, MAE: {:.4f}, MSE: {:.4f}\n'.format(phase, epoch_loss, epoch_mae, epoch_mse))
                        f.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                        f.close()
                print('{}: Loss: {:.4f}, MAE: {:.4f} MSE: {:.4f}'.format(phase, epoch_loss, epoch_mae, epoch_mse))

    def run(self):
        self.train_model(self.model, self.optimizer, self.num_epoch, self.val_size)
