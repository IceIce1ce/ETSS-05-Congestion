import os
import numpy as np
import json
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from model import CSRNet as net_PSL
import dataset
import utils
import argparse
import warnings
warnings.filterwarnings("ignore")

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

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def main(args):
    best_prec1 = 1e6
    train_list = json.load(open(args.train_json))
    val_list = json.load(open(args.val_json))
    torch.cuda.manual_seed(args.seed)
    # model
    model = net_PSL()
    # model = nn.DataParallel(model)
    model = model.cuda()
    # loss
    criterion = nn.MSELoss(size_average=False).cuda()
    if args.opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
    elif args.opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)
    else:
        print('This optimizer does not exist')
        raise NotImplementedError
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    best_txt = os.path.join(args.output_dir, 'best.txt')
    # train and test loader
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=True, args=args), batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False, args=args), batch_size=1)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        prec1, prec_net1, prec_net2, mse, mse_net1, mse_net2 = validate(val_loader, model)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print('Best MAE: {:.4f}'.format(best_prec1))
        with open(best_txt, 'a+') as txtfile:
            txtfile.write("Epoch: {}, MAE: {:.4f}, Net1: {:.4f}, Net2: {:.4f}, MSE: {:.4f}, Net1: {:.4f}, Net2: {:.4f}\n".format(epoch, prec1,prec_net1, prec_net2, mse, mse_net1, mse_net2))
        if is_best:
            with open(best_txt, 'a+') as txtfile:
                txtfile.write("Best epoch: {}, MAE: {:.4f}, Net1: {:.4f}, Net2: {:.4f}, MSE: {:.4f}, Net1: {:.4f}, Net2: {:.4f}\n".format(epoch, best_prec1,prec_net1, prec_net2, mse, mse_net1, mse_net2))
            utils.save_checkpoint({'args': args, 'epoch': epoch, 'model': model, 'best_result': best_prec1, 'optimizer': optimizer, 'state': model.state_dict()}, is_best, epoch, args.output_dir)
    
def train(train_loader, model, criterion, optimizer, epoch):
    criterion_lcm = nn.L1Loss().cuda()
    losses = AverageMeter()
    model.train()
    for i, (img, target, mask, target_20) in enumerate(train_loader):
        mask[mask > 0] = 1 # [1, 60, 28]
        mask = mask.cuda() # [1, 60, 28]
        mask = mask.permute([0, 2, 1]) # [1, 28, 60]
        target = target.type(torch.FloatTensor).cuda() # [1, 28, 60]
        target = Variable(target) # [1, 28, 60]
        target_20 = target_20.type(torch.FloatTensor).cuda() # [1, 28, 60]
        target_20 = Variable(target_20) # [1, 28, 60]
        target = target * mask # [1, 28, 60]
        target_20 = target_20 * mask # [1, 28, 60]
        img = img.cuda() # [1, 3, 225, 485]
        img = Variable(img) # [1, 3, 225, 485]
        with torch.no_grad():
            output_1, output_2, _, _, _, _ = model(img, mask=None, target=None, train_flag=False) # [1, 1, 21, 38], [1, 1, 21, 38]
            target_pred = (output_1 + output_2) / 2 # [1, 1, 21, 38]
        output_1, output_2, latent_loss, _, diff_mean, diff_var = model(img, mask, target_pred) # [1, 1, 21, 38], [1, 1, 21, 38]
        # net 1
        output_net1 = output_1[:, 0, :, :] # [1, 34, 64]
        output_net1 = output_net1 * mask # [1, 34, 64]
        pred_loss_net1 = criterion(output_net1, target)
        # net 2
        output_net2 = output_2[:, 0, :, :] # [1, 34, 64]
        output_net2 = output_net2 * mask # [1, 34, 64]
        pred_loss_net2 = criterion(output_net2, target_20)
        output_net1_sum = torch.sum(output_1,dim=[1, 2, 3]) # [1]
        output_net2_sum = torch.sum(output_2, dim=[1, 2, 3]) # [1]
        pred_loss_sum_net1_net2 = criterion_lcm(output_net1_sum, output_net2_sum.detach())
        pred_loss_sum_net2_net1 = criterion_lcm(output_net2_sum, output_net1_sum.detach())
        pred_loss = pred_loss_net1 + pred_loss_net2 + 0.1 * pred_loss_sum_net1_net2 + 0.1 * pred_loss_sum_net2_net1
        latent_loss_weight = 0.1
        latent_loss = latent_loss.mean()
        loss = pred_loss + latent_loss_weight * latent_loss
        dis_loss_weight = args.lambda_u * linear_rampup(epoch + i / (len(train_loader)), args.epochs)
        mean_ann, mean_unkn = diff_mean[0], diff_mean[1]
        var_ann, var_unkn = diff_var[0], diff_var[1]
        loss_mean = criterion(mean_ann.detach(), mean_unkn)
        loss_var = criterion(var_ann.detach(), var_unkn)
        dis_loss = loss_mean + loss_var
        loss = loss + dis_loss_weight * dis_loss
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

def validate(val_loader, model):
    model.eval()
    mae, mse = 0, 0
    mae_net1, mse_net1 = 0, 0
    mae_net2, mse_net2 = 0, 0
    for i, (img, target, mask, _) in enumerate(val_loader): # [1, 3, 768, 1024], [1, 96, 128], [1, 196, 128]
        h, w = img.shape[2:4]
        h_d = h // 2
        w_d = w // 2
        img_1 = Variable(img[:, :, :h_d, :w_d].cuda())
        img_2 = Variable(img[:, :, :h_d, w_d:].cuda())
        img_3 = Variable(img[:, :, h_d:, :w_d].cuda())
        img_4 = Variable(img[:, :, h_d:, w_d:].cuda())
        density_1, density_net2_1, _, _,_,_ = model(img_1)
        density_2, density_net2_2, _, _,_,_ = model(img_2)
        density_3, density_net2_3, _, _,_,_ = model(img_3)
        density_4, density_net2_4, _, _,_,_ = model(img_4)
        density_1 = density_1.data.cpu().numpy()
        density_2 = density_2.data.cpu().numpy()
        density_3 = density_3.data.cpu().numpy()
        density_4 = density_4.data.cpu().numpy()
        pred_sum_1 = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
        density_net2_1 = density_net2_1.data.cpu().numpy()
        density_net2_2 = density_net2_2.data.cpu().numpy()
        density_net2_3 = density_net2_3.data.cpu().numpy()
        density_net2_4 = density_net2_4.data.cpu().numpy()
        pred_sum_net2 = density_net2_1.sum() + density_net2_2.sum() + density_net2_3.sum() + density_net2_4.sum()
        pred_sum = (pred_sum_1 + pred_sum_net2) / 2
        mae_net1 += abs(pred_sum_1 - target.sum())
        mae_net2 += abs(pred_sum_net2 - target.sum())
        mse_net1 += (pred_sum_1 - target.sum()) ** 2
        mse_net2 += (pred_sum_net2 - target.sum()) ** 2
        mae += abs(pred_sum - target.sum())
        mse += (pred_sum - target.sum()) ** 2
    mae = mae / len(val_loader)
    mse = mse / len(val_loader)
    mae_net1 = mae_net1 / len(val_loader)
    mse_net1 = mse_net1 / len(val_loader)
    mae_net2 = mae_net2 / len(val_loader)
    mse_net2 = mse_net2 / len(val_loader)
    print('MAE: {mae:.4f}, MAE net1: {mae_net1:.4f}, MAE net2: {mae_net2:.4f}, MSE {mse:.4f}'.format(mae=mae, mae_net1=mae_net1, mae_net2=mae_net2, mse=mse))
    return mae, mae_net1, mae_net2, np.sqrt(mse), np.sqrt(mse_net1), np.sqrt(mse_net2)

def eval(args):
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)
    torch.cuda.manual_seed(args.seed)
    # model
    model = net_PSL()
    # model = nn.DataParallel(model)
    model = model.cuda()
    # test loader
    val_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False, args=args), batch_size=1)
    # test
    for epoch in range(args.epoch_st, args.epoch_end):
        ckpt = torch.load(os.path.join(args.output_dir, 'model_best.pth.tar'))
        model.load_state_dict(ckpt['state'])
        model.eval()
        mae, mse = 0, 0
        mae_net1, mse_net1 = 0, 0
        mae_net2, mse_net2 = 0, 0
        for i, (img, target, mask, _, img_path) in enumerate(val_loader):
            h,w = img.shape[2:4]
            h_d = h // 2
            w_d = w // 2
            img_1 = Variable(img[:, :, :h_d, :w_d].cuda())
            img_2 = Variable(img[:, :, :h_d, w_d:].cuda())
            img_3 = Variable(img[:, :, h_d:, :w_d].cuda())
            img_4 = Variable(img[:, :, h_d:, w_d:].cuda())
            density_1, density_net2_1, _, _,_,_ = model(img_1)
            density_2, density_net2_2, _, _,_,_ = model(img_2)
            density_3, density_net2_3, _, _,_,_ = model(img_3)
            density_4, density_net2_4, _, _,_,_ = model(img_4)
            density_1 = density_1.data.cpu().numpy()
            density_2 = density_2.data.cpu().numpy()
            density_3 = density_3.data.cpu().numpy()
            density_4 = density_4.data.cpu().numpy()
            pred_sum_1 = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
            density_net2_1 = density_net2_1.data.cpu().numpy()
            density_net2_2 = density_net2_2.data.cpu().numpy()
            density_net2_3 = density_net2_3.data.cpu().numpy()
            density_net2_4 = density_net2_4.data.cpu().numpy()
            pred_sum_net2 = density_net2_1.sum() + density_net2_2.sum() + density_net2_3.sum() + density_net2_4.sum()
            pred_sum = (pred_sum_1 + pred_sum_net2) / 2
            mae_net1 += abs(pred_sum_1 - target.sum())
            mae_net2 += abs(pred_sum_net2 - target.sum())
        mse_net1 += (pred_sum_1 - target.sum()) ** 2
        mse_net2 += (pred_sum_net2 - target.sum()) ** 2
        mae += abs(pred_sum - target.sum())
        mse += (pred_sum - target.sum()) ** 2
    mae = mae / len(val_loader)
    mse = mse / len(val_loader)
    mae_net1 = mae_net1 / len(val_loader)
    mae_net2 = mae_net2 / len(val_loader)
    print('MAE: {mae:.4f}, MAE net1: {mae_net1:.4f}, MAE net2: {mae_net2:.4f}, MSE: {mse:.4f}'.format(mae=mae, mae_net1=mae_net1, mae_net2=mae_net2, mse=mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--type_dataset', type=str, default="sha", choices=['sha', 'shb'])
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--seed', default=time.time(), type=int)
    parser.add_argument('--train_json', default='A_train.json', type=str)
    parser.add_argument('--val_json', default='A_val.json', type=str)
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    # model config
    parser.add_argument('--arch', default='CSRNet', type=str)
    parser.add_argument('--decoder', default='upproj', type=str)
    # training config
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_patience', default=2, type=int)
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--gt_ratio', default='2020_12_16_v2_rand_', type=str)
    parser.add_argument('--opt_type', default='Adam', type=str, choices=['Adam', 'SGD'])
    parser.add_argument('--epoch_st', default=10, type=int)
    parser.add_argument('--epoch_end', default=20, type=int)
    parser.add_argument('--decay', type=float, default=5 * 1e-4)
    # data augmentation config
    parser.add_argument('--use_random_crop', action='store_true', default=False)
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u', default=75, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    # testing config
    parser.add_argument('--is_eval', action='store_true', default=False)
    args = parser.parse_args()

    if args.is_eval:
        print('Testing dataset:', args.type_dataset)
        eval(args)
    else:
        print('Training dataset:', args.type_dataset)
        main(args)