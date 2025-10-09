from model import CANNet2s
from utils import save_checkpoint
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import argparse
import json
import dataset
import time
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

def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # train loader
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list, shuffle=True, transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=True,
                                               batch_size=args.batch_size, num_workers=args.num_workers), batch_size=args.batch_size)
    model.train()
    end = time.time()
    for i, (prev_img, img, post_img, prev_target, target, post_target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        prev_img = prev_img.cuda() # [1, 3, 360, 640]
        prev_img = Variable(prev_img)
        img = img.cuda() # [1, 3, 360, 640]
        img = Variable(img)
        post_img = post_img.cuda() # [1, 3, 360, 640]
        post_img = Variable(post_img)
        prev_flow = model(prev_img, img) # [1, 10, 45, 80]
        post_flow = model(img,post_img) # [1, 10, 45, 80]
        prev_flow_inverse = model(img ,prev_img) # [1, 10, 45, 80]
        post_flow_inverse = model(post_img, img) # [1, 10, 45, 80]
        target = target.type(torch.FloatTensor)[0].cuda() # [45, 80]
        target = Variable(target)
        prev_target = prev_target.type(torch.FloatTensor)[0].cuda() # [45, 80]
        prev_target = Variable(prev_target)
        post_target = post_target.type(torch.FloatTensor)[0].cuda() # [45, 80]
        post_target = Variable(post_target)
        mask_boundry = torch.zeros(prev_flow.shape[2:]) # [45, 80]
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0
        mask_boundry = Variable(mask_boundry.cuda())
        reconstruction_from_prev = F.pad(prev_flow[0, 0, 1:, 1:],(0, 1, 0, 1)) + F.pad(prev_flow[0, 1, 1:, :],(0, 0, 0, 1)) +\
                                   F.pad(prev_flow[0, 2, 1:, :-1],(1, 0, 0, 1)) + F.pad(prev_flow[0, 3, :, 1:],(0, 1, 0, 0)) +\
                                   prev_flow[0, 4, :, :] + F.pad(prev_flow[0, 5, :, :-1],(1, 0, 0, 0)) + F.pad(prev_flow[0, 6, :-1, 1:],(0, 1, 1, 0)) +\
                                   F.pad(prev_flow[0, 7, :-1, :],(0, 0, 1, 0)) + F.pad(prev_flow[0, 8, :-1, :-1],(1, 0, 1, 0)) + prev_flow[0, 9, :, :] * mask_boundry # [45, 80]
        reconstruction_from_post = torch.sum(post_flow[0, :9, :, :], dim=0) + post_flow[0, 9, :, :] * mask_boundry # [45, 80]
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :, :] * mask_boundry # [45, 80]
        reconstruction_from_post_inverse = F.pad(post_flow_inverse[0, 0, 1:, 1:],(0, 1, 0, 1)) + F.pad(post_flow_inverse[0, 1, 1:, :],(0, 0, 0, 1)) +\
                                           F.pad(post_flow_inverse[0,2,1:,:-1],(1,0,0,1)) + F.pad(post_flow_inverse[0,3,:,1:],(0,1,0,0)) +\
                                           post_flow_inverse[0,4,:,:] + F.pad(post_flow_inverse[0,5,:,:-1],(1,0,0,0)) + F.pad(post_flow_inverse[0,6,:-1,1:],(0,1,1,0)) +\
                                           F.pad(post_flow_inverse[0,7,:-1,:],(0,0,1,0)) + F.pad(post_flow_inverse[0,8,:-1,:-1],(1,0,1,0)) +\
                                           post_flow_inverse[0,9,:,:] * mask_boundry # [45, 80]
        prev_reconstruction_from_prev = torch.sum(prev_flow[0,:9,:,:],dim=0) + prev_flow[0,9,:,:] * mask_boundry # [45, 80]
        post_reconstruction_from_post = F.pad(post_flow[0,0,1:,1:],(0,1,0,1)) + F.pad(post_flow[0,1,1:,:],(0,0,0,1)) + F.pad(post_flow[0,2,1:,:-1],(1,0,0,1)) +\
                                        F.pad(post_flow[0,3,:,1:],(0,1,0,0)) + post_flow[0,4,:,:] + F.pad(post_flow[0,5,:,:-1],(1,0,0,0)) +\
                                        F.pad(post_flow[0,6,:-1,1:],(0,1,1,0)) + F.pad(post_flow[0,7,:-1,:],(0,0,1,0)) + F.pad(post_flow[0,8,:-1,:-1],(1,0,1,0)) +\
                                        post_flow[0,9,:,:] * mask_boundry # [45, 80]
        loss_prev_flow = criterion(reconstruction_from_prev, target)
        loss_post_flow = criterion(reconstruction_from_post, target)
        loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
        loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)
        loss_prev = criterion(prev_reconstruction_from_prev,prev_target)
        loss_post = criterion(post_reconstruction_from_post,post_target)
        loss_prev_consistency = criterion(prev_flow[0,0,1:,1:], prev_flow_inverse[0,8,:-1,:-1]) + criterion(prev_flow[0,1,1:,:], prev_flow_inverse[0,7,:-1,:]) +\
                                criterion(prev_flow[0,2,1:,:-1], prev_flow_inverse[0,6,:-1,1:]) + criterion(prev_flow[0,3,:,1:], prev_flow_inverse[0,5,:,:-1]) +\
                                criterion(prev_flow[0,4,:,:], prev_flow_inverse[0,4,:,:]) + criterion(prev_flow[0,5,:,:-1], prev_flow_inverse[0,3,:,1:]) +\
                                criterion(prev_flow[0,6,:-1,1:], prev_flow_inverse[0,2,1:,:-1]) + criterion(prev_flow[0,7,:-1,:], prev_flow_inverse[0,1,1:,:]) +\
                                criterion(prev_flow[0,8,:-1,:-1], prev_flow_inverse[0,0,1:,1:])
        loss_post_consistency = criterion(post_flow[0,0,1:,1:], post_flow_inverse[0,8,:-1,:-1]) + criterion(post_flow[0,1,1:,:], post_flow_inverse[0,7,:-1,:]) +\
                                criterion(post_flow[0,2,1:,:-1], post_flow_inverse[0,6,:-1,1:]) + criterion(post_flow[0,3,:,1:], post_flow_inverse[0,5,:,:-1]) +\
                                criterion(post_flow[0,4,:,:], post_flow_inverse[0,4,:,:]) + criterion(post_flow[0,5,:,:-1], post_flow_inverse[0,3,:,1:]) +\
                                criterion(post_flow[0,6,:-1,1:], post_flow_inverse[0,2,1:,:-1]) + criterion(post_flow[0,7,:-1,:], post_flow_inverse[0,1,1:,:]) +\
                                criterion(post_flow[0,8,:-1,:-1], post_flow_inverse[0,0,1:,1:])
        loss = loss_prev_flow + loss_post_flow + loss_prev_flow_inverse + loss_post_flow_inverse + loss_prev + loss_post + loss_prev_consistency + loss_post_consistency
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], Time {batch_time.val:.3f} ({batch_time.avg:.3f}), Data {data_time.val:.3f} ({data_time.avg:.3f}), Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))

def validate(val_list, model):
    # test loader
    val_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, shuffle=False, transform=transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=False), batch_size=1)
    model.eval()
    mae = 0
    for i, (prev_img, img, post_img, _, target, _) in enumerate(val_loader):
        prev_img = prev_img.cuda() # [1, 3, 360, 640]
        prev_img = Variable(prev_img)
        img = img.cuda() # [1, 3, 360, 640]
        img = Variable(img)
        prev_flow = model(prev_img, img) # [1, 10, 45, 80]
        prev_flow_inverse = model(img, prev_img) # [1, 10, 45, 80]
        target = target.type(torch.FloatTensor)[0].cuda() # [45, 80]
        target = Variable(target)
        mask_boundry = torch.zeros(prev_flow.shape[2:]) # [45, 80]
        mask_boundry[0,:] = 1.0
        mask_boundry[-1,:] = 1.0
        mask_boundry[:,0] = 1.0
        mask_boundry[:,-1] = 1.0
        mask_boundry = Variable(mask_boundry.cuda())
        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1)) + F.pad(prev_flow[0,1,1:,:],(0,0,0,1)) + F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1)) +\
                                   F.pad(prev_flow[0,3,:,1:],(0,1,0,0)) + prev_flow[0,4,:,:] + F.pad(prev_flow[0,5,:,:-1],(1,0,0,0)) +\
                                   F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0)) + F.pad(prev_flow[0,7,:-1,:],(0,0,1,0)) + F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0)) +\
                                   prev_flow[0,9,:,:] * mask_boundry # [45, 80]
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0) + prev_flow_inverse[0,9,:,:] * mask_boundry # [45, 80]
        overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0).type(torch.FloatTensor) # [45, 80]
        target = target.type(torch.FloatTensor)
        mae += abs(overall.data.sum() - target.sum())
    mae = mae / len(val_loader)
    print('MAE: {:.4f}'.format(mae))
    return mae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_json', type=str, default='datasets/train.json')
    parser.add_argument('val_json', type=str, default='datasets/val.json')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--decay', type=float, default=5 * 1e-4)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--print_freq', type=int, default=1000)
    args = parser.parse_args()

    best_prec1 = 1e6
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)
    torch.cuda.manual_seed(args.seed)
    # model
    model = CANNet2s()
    model = model.cuda()
    # loss
    criterion = nn.MSELoss(size_average=False).cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
    for epoch in range(args.start_epoch, args.epochs):
        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print('Besat MAE: {:.4f}'.format(best_prec1))
        save_checkpoint({'state_dict': model.state_dict()}, is_best)
