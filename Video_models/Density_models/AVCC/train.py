import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import math
import time
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, cm
from torch import nn
from torch.autograd import Variable
import albumentations as A
from torchvision import transforms
import yaml
import dataset
from utils import save_checkpoint
from variables import MEAN, STD, PATCH_SIZE_PF
from model import XACANNet2s, CANNet2s

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plotDensity(density, axarr, k):
    density = density * 255.0
    colormap_i = cm.jet(density)[:, :, 0:3]
    overlay_i = colormap_i
    new_map = overlay_i.copy()
    new_map[:, :, 0] = overlay_i[:, :, 2]
    new_map[:, :, 2] = overlay_i[:, :, 0]
    axarr[k].imshow(255 * new_map.astype(np.uint8))

class Criterion(nn.Module):
    def __init__(self, uncertainty=False):
        super().__init__()
        self.uncertainty = uncertainty
        if uncertainty:
            self.base_var = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x, target, unc=True):
        if self.uncertainty and unc:
            mean, var = x[0, ...], x[1, ...]
            var = var + self.base_var
            nonzero_var_idxs = (var != 0)
            loss1 = ((1 / var[nonzero_var_idxs]) * (mean[nonzero_var_idxs] - target[nonzero_var_idxs]) ** 2).sum()
            loss2 = torch.log(var[nonzero_var_idxs]).sum()
            return 0.5 * (loss1 + loss2)
        else:
            return F.mse_loss(x, target, reduction='sum')

def compute_densities_from_flows(prev_flow, post_flow, prev_flow_inverse, post_flow_inverse):
    mask_boundry = torch.zeros(prev_flow.shape[2:])
    mask_boundry[0, :] = 1.0
    mask_boundry[-1, :] = 1.0
    mask_boundry[:, 0] = 1.0
    mask_boundry[:, -1] = 1.0
    mask_boundry = Variable(mask_boundry.cuda())
    reconstruction_from_prev = F.pad(prev_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(prev_flow[0, 1, 1:, :], (0, 0, 0, 1)) + \
                               F.pad(prev_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(prev_flow[0, 3, :, 1:], (0, 1, 0, 0)) + prev_flow[0, 4, :, :] + \
                               F.pad(prev_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + \
                               F.pad(prev_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + prev_flow[0, 9, :, :] * mask_boundry
    reconstruction_from_post = torch.sum(post_flow[0, :9, :, :], dim=0) + post_flow[0, 9, :, :] * mask_boundry
    reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :, :] * mask_boundry
    reconstruction_from_post_inverse = F.pad(post_flow_inverse[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(post_flow_inverse[0, 1, 1:, :], (0, 0, 0, 1)) + \
                                       F.pad(post_flow_inverse[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(post_flow_inverse[0, 3, :, 1:], (0, 1, 0, 0)) + post_flow_inverse[0, 4, :, :] + \
                                       F.pad(post_flow_inverse[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(post_flow_inverse[0, 6, :-1, 1:], (0, 1, 1, 0)) + \
                                       F.pad(post_flow_inverse[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(post_flow_inverse[0, 8, :-1, :-1], (1, 0, 1, 0)) + post_flow_inverse[0, 9, :, :] * mask_boundry
    prev_reconstruction_from_prev = torch.sum(prev_flow[0, :9, :, :], dim=0) + prev_flow[0, 9, :, :] * mask_boundry
    post_reconstruction_from_post = F.pad(post_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(post_flow[0, 1, 1:, :], (0, 0, 0, 1)) + \
                                    F.pad(post_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(post_flow[0, 3, :, 1:], (0, 1, 0, 0)) + post_flow[0, 4, :, :] + \
                                    F.pad(post_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(post_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + \
                                    F.pad(post_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(post_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + post_flow[0, 9, :, :] * mask_boundry
    return reconstruction_from_prev, reconstruction_from_post, reconstruction_from_prev_inverse, reconstruction_from_post_inverse, prev_reconstruction_from_prev, post_reconstruction_from_post

def train(config, train_list, val_list, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    alb_transforms = A.Compose([A.RandomResizedCrop(config['height'], config['width'], scale=(0.75, 1.0), ratio=(0.95, 1.05), p=0.5), A.HorizontalFlip(p=0.5),
                                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), A.Normalize(mean=MEAN, std=STD)],
                               additional_targets={'image1': 'image', 'image2': 'image', 'density': 'mask', 'density1': 'mask', 'density2': 'mask', 'mask': 'mask'})
    # train loader
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list, shuffle=True, transform=alb_transforms, train=True, shape=(config['width'], config['height']), mask=None), batch_size=args.batch_size)
    model.train()
    optimizer.zero_grad()
    for i, (prev_img, img, post_img, prev_target, target, post_target, frame_mask) in enumerate(train_loader):
        prev_img = prev_img.cuda() # [1, 3, 360, 640]
        prev_img = Variable(prev_img)
        img = img.cuda() # [1, 3, 360, 640]
        img = Variable(img)
        post_img = post_img.cuda() # [1, 3, 360, 640]
        post_img = Variable(post_img)
        prev_flow = model(prev_img, img) # [1, 10, 45, 80]
        post_flow = model(img, post_img) # [1, 10, 45, 80]
        prev_flow_inverse = model(img, prev_img) # [1, 10, 45, 80]
        post_flow_inverse = model(post_img, img) # [1, 10, 45, 80]
        target = target.type(torch.FloatTensor)[0].cuda() # [45, 80]
        target = Variable(target)
        prev_target = prev_target.type(torch.FloatTensor)[0].cuda() # [45, 80]
        prev_target = Variable(prev_target)
        post_target = post_target.type(torch.FloatTensor)[0].cuda() # [45, 80]
        post_target = Variable(post_target)
        (reconstruction_from_prev, reconstruction_from_post, reconstruction_from_prev_inverse, reconstruction_from_post_inverse, prev_reconstruction_from_prev,
         post_reconstruction_from_post) = compute_densities_from_flows(prev_flow[:, :10, :, :], post_flow[:, :10, :, :], prev_flow_inverse[:, :10, :, :], post_flow_inverse[:, :10, :, :]) # [45, 80]
        if args.uncertainty:
            (reconstruction_from_prev_var, reconstruction_from_post_var, reconstruction_from_prev_inverse_var, reconstruction_from_post_inverse_var, prev_reconstruction_from_prev_var,
             post_reconstruction_from_post_var) = compute_densities_from_flows(prev_flow[:, 10:, :, :], post_flow[:, 10:, :, :], prev_flow_inverse[:, 10:, :, :], post_flow_inverse[:, 10:, :, :])
            reconstruction_from_prev = torch.stack([reconstruction_from_prev, reconstruction_from_prev_var]) # [2, 45, 80]
            reconstruction_from_post = torch.stack([reconstruction_from_post, reconstruction_from_post_var]) # [2, 45, 80]
            reconstruction_from_prev_inverse = torch.stack([reconstruction_from_prev_inverse, reconstruction_from_prev_inverse_var]) # [2, 45, 80]
            reconstruction_from_post_inverse = torch.stack([reconstruction_from_post_inverse, reconstruction_from_post_inverse_var]) # [2, 45, 80]
            prev_reconstruction_from_prev = torch.stack([prev_reconstruction_from_prev, prev_reconstruction_from_prev_var]) # [2, 45, 80]
            post_reconstruction_from_post = torch.stack([post_reconstruction_from_post, post_reconstruction_from_post_var]) # [2, 45, 80]
        loss_prev_flow = criterion(reconstruction_from_prev, target)
        loss_post_flow = criterion(reconstruction_from_post, target)
        loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
        loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)
        loss_prev = criterion(prev_reconstruction_from_prev, prev_target)
        loss_post = criterion(post_reconstruction_from_post, post_target)
        loss_prev_consistency = criterion(prev_flow[0, 0, 1:, 1:], prev_flow_inverse[0, 8, :-1, :-1], unc=False) + \
                                criterion(prev_flow[0, 1, 1:, :], prev_flow_inverse[0, 7, :-1, :], unc=False) + \
                                criterion(prev_flow[0, 2, 1:, :-1], prev_flow_inverse[0, 6, :-1, 1:], unc=False) + \
                                criterion(prev_flow[0, 3, :, 1:], prev_flow_inverse[0, 5, :, :-1], unc=False) + \
                                criterion(prev_flow[0, 4, :, :], prev_flow_inverse[0, 4, :, :], unc=False) + \
                                criterion(prev_flow[0, 5, :, :-1], prev_flow_inverse[0, 3, :, 1:], unc=False) + \
                                criterion(prev_flow[0, 6, :-1, 1:], prev_flow_inverse[0, 2, 1:, :-1], unc=False) + \
                                criterion(prev_flow[0, 7, :-1, :], prev_flow_inverse[0, 1, 1:, :], unc=False) + \
                                criterion(prev_flow[0, 8, :-1, :-1], prev_flow_inverse[0, 0, 1:, 1:], unc=False)
        loss_post_consistency = criterion(post_flow[0, 0, 1:, 1:], post_flow_inverse[0, 8, :-1, :-1], unc=False) + \
                                criterion(post_flow[0, 1, 1:, :], post_flow_inverse[0, 7, :-1, :], unc=False) + \
                                criterion(post_flow[0, 2, 1:, :-1], post_flow_inverse[0, 6, :-1, 1:], unc=False) + \
                                criterion(post_flow[0, 3, :, 1:], post_flow_inverse[0, 5, :, :-1], unc=False) + \
                                criterion(post_flow[0, 4, :, :], post_flow_inverse[0, 4, :, :], unc=False) + \
                                criterion(post_flow[0, 5, :, :-1], post_flow_inverse[0, 3, :, 1:], unc=False) + \
                                criterion(post_flow[0, 6, :-1, 1:], post_flow_inverse[0, 2, 1:, :-1], unc=False) + \
                                criterion(post_flow[0, 7, :-1, :], post_flow_inverse[0, 1, 1:, :], unc=False) + \
                                criterion(post_flow[0, 8, :-1, :-1], post_flow_inverse[0, 0, 1:, 1:], unc=False)
        loss = loss_prev_flow + loss_post_flow + loss_prev_flow_inverse + loss_post_flow_inverse + loss_prev + loss_post + loss_prev_consistency + loss_post_consistency
        losses.update(loss.item(), img.size(0))
        loss.backward()
        if (i + 1) % args.virtual_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        if (i + 1) % config['log_freq'] == 0:
            if len(reconstruction_from_prev.shape) == 3:
                reconstruction_from_prev = reconstruction_from_prev[0]
                reconstruction_from_prev_inverse = reconstruction_from_prev_inverse[0]
            overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0).data.cpu().numpy()
            pred = cv2.resize(overall, (overall.shape[1] * PATCH_SIZE_PF, overall.shape[0] * PATCH_SIZE_PF), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2) # [360, 640
            target = cv2.resize(target.cpu().detach().numpy(), (target.shape[1] * PATCH_SIZE_PF, target.shape[0] * PATCH_SIZE_PF), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2) # [360, 640]
            if args.is_vis:
                fig, axarr = plt.subplots(1, 2)
                plotDensity(pred, axarr, 0)
                plotDensity(target, axarr, 1)
                if not os.path.exists(args.vis_dir):
                    os.makedirs(args.vis_dir)
                fig.savefig(f'{args.vis_dir}/epoch_{epoch + 1}.png', dpi=100, bbox_inches='tight')
            print('Epoch: [{0}][{1}/{2}], Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(epoch + 1, i + 1, len(train_loader), loss=losses))
        if ((i + 1) % config['log_freq'] == 0) & ((i + 1) != len(train_loader)):
            prec1 = validate(config, val_list, model)
            is_best = prec1 < args.best_prec1
            args.best_prec1 = min(prec1, args.best_prec1)
            print('Best MAE: {:.2f}'.format(args.best_prec1))
            save_checkpoint({'epoch': epoch, 'start_frame': i + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_prec': args.best_prec1,
                             'config': config}, is_best, epoch, args)

def validate(config, val_list, model):
    # test loader
    val_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, shuffle=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]),
                                                                 train=False, shape=(config['width'], config['height']), mask=None))
    model.eval()
    mae = 0.0
    mse = 0.0
    for i, (prev_img, img, post_img, _, target, _, frame_mask) in enumerate(val_loader):
        prev_img = prev_img.cuda() # [1, 3, 360, 640]
        prev_img = Variable(prev_img)
        img = img.cuda() # [1, 3, 360, 640]
        img = Variable(img)
        with torch.no_grad():
            prev_flow = model(prev_img, img)[:, :10, ...] # [1, 10, 45, 80]
            prev_flow_inverse = model(img, prev_img)[:, :10, ...] # [1, 10, 45, 80]
        target = target.type(torch.FloatTensor)[0].cuda() # [45, 80]
        target = Variable(target)
        mask_boundry = torch.zeros(prev_flow.shape[2:]) # [45, 80]
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0
        mask_boundry = Variable(mask_boundry.cuda())
        reconstruction_from_prev = F.pad(prev_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(prev_flow[0, 1, 1:, :], (0, 0, 0, 1)) + \
                                   F.pad(prev_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(prev_flow[0, 3, :, 1:], (0, 1, 0, 0)) + prev_flow[0, 4, :, :] + \
                                   F.pad(prev_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + \
                                   F.pad(prev_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + prev_flow[0, 9, :, :] * mask_boundry # [45, 80]
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :, :] * mask_boundry # [45, 80]
        overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0) # [45, 80]
        overall = overall.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        if i % config['log_freq'] == 0:
            print('Pred: {:.2f}, GT: {:.2f}'.format(overall.data.sum(), target.sum()))
        mae += abs(overall.data.sum() - target.sum())
        mse += abs(overall.data.sum() - target.sum()) * abs(overall.data.sum() - target.sum())
    mae = mae / len(val_loader)
    mse = math.sqrt(mse / len(val_loader))
    print('MAE: {:.2f}, MSE: {:.2f}'.format(mae, mse))
    return mae

def main(args):
    # load config
    with open(args.cfg, 'r') as ymlfile:
        configs = yaml.safe_load(ymlfile)
    with open(configs['train_json'], 'r') as outfile:
        train_list = json.load(outfile)
    with open(configs['val_json'], 'r') as outfile:
        val_list = json.load(outfile)
    args.uncertainty = True if 'uncertainty' in configs and configs['uncertainty'] else False
    torch.cuda.manual_seed(args.seed)
    # model
    model_fn = eval(configs['model'])
    model = model_fn(load_weights=False, uncertainty=args.uncertainty) # SACANNet2s(load_weights=False, fine_tuning=False)
    model = model.cuda()
    # loss
    criterion = Criterion(uncertainty=args.uncertainty)
    criterion = criterion.cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), configs['lr'], weight_decay=args.decay)
    try:
        model_dir = os.path.join(args.output_dir, 'best.pth.tar')
        checkpoint = torch.load(model_dir, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        args.start_frame = checkpoint['start_frame']
        try:
            args.best_prec1 = checkpoint['best_prec'].item()
        except:
            args.best_prec1 = checkpoint['best_prec']
        print('Load ckpt from:', model_dir)
    except:
        print('Start training from scratch')
    for epoch in range(args.start_epoch, args.epochs):
        train(configs, train_list, val_list, model, criterion, optimizer, epoch, args)
        prec1 = validate(configs, val_list, model)
        is_best = prec1 < args.best_prec1
        args.best_prec1 = min(prec1, args.best_prec1)
        args.start_frame = 0
        print('Best MAE: {:.2f} '.format(args.best_prec1))
        save_checkpoint({'epoch': epoch + 1, 'start_frame': 0, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_prec': args.best_prec1,
                         'config': configs}, is_best, epoch, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='FDST')
    parser.add_argument('--cfg', type=str, default='configs/fdst_XA.yaml')
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--output_dir', type=str, default='saved_fdst')
    parser.add_argument('--is_vis', action='store_true')
    parser.add_argument('--vis_dir', type=str, default='vis_fdst')
    # training config
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--virtual_batch_size', type=int, default=10)
    parser.add_argument('--best_prec1', type=int, default=1e6)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--decay', type=float, default=5 * 1e-4)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    main(args)