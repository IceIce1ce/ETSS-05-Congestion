import os
from model import CSRNet
from utils import save_checkpoint, adjust_learning_rate, AverageMeter
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import argparse
import json
import dataset
import time
import warnings
warnings.filterwarnings("ignore")

def train(train_list, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list, shuffle=True, transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), train=True, seen=model.seen,
                                               batch_size=args.batch_size, num_workers=args.num_workers), batch_size=args.batch_size)
    model.train()
    for i, (img, target) in enumerate(train_loader):
        img = img.cuda() # [1, 3, 600, 900]
        img = Variable(img)
        output = model(img) # [1, 1, 75, 112]
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda() # [1, 1, 75, 112]
        target = Variable(target)
        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.print_freq == 0:
            print('Epoch: [{}/{}], Iter: [{}/{}], Loss: [{:.4f}/{:.4f}]'.format(epoch, args.epochs, i, len(train_loader), losses.val, losses.avg))
    
def validate(val_list, model):
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, shuffle=False, transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  train=False), batch_size=args.batch_size)
    model.eval()
    mae, mse = 0, 0
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda() # [1, 3, 615, 1024]
        img = Variable(img)
        output = model(img) # [1, 1, 76, 1028]
        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
        mse += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()) ** 2
    mae = mae / len(test_loader)
    mse = mse / len(test_loader)
    print('MAE {:.4f}, RMSE: {:.4f}'.format(mae, mse))
    return mae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha', choices=['sha', 'shb'])
    parser.add_argument('--train_json', default='configs/part_A_train.json', type=str)
    parser.add_argument('--test_json', default='configs/part_A_test.json', type=str)
    parser.add_argument('--ckpt_dir', default=None, type=str)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--original_lr', type=float, default=1e-7)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--decay', type=float, default=5 * 1e-4)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=30)
    args = parser.parse_args()
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.seed = time.time()

    print('Training dataset:', args.type_dataset)
    best_prec1 = 1e6
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)
    torch.cuda.manual_seed(args.seed)
    # model
    model = CSRNet()
    model = model.cuda()
    # loss
    criterion = nn.MSELoss(size_average=False).cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)
    # resume training
    if args.ckpt_dir is not None:
        if os.path.isfile(args.ckpt_dir):
            checkpoint = torch.load(args.ckpt_dir)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Load ckpt from: {}".format(args.ckpt_dir))
        else:
            print("No ckpt found at: {}".format(args.ckpt_dir))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_list, model, criterion, optimizer, epoch, args)
        prec1 = validate(val_list, model)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print('Best MAE: {mae:.4f}'.format(mae=best_prec1))
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_prec1, 'optimizer': optimizer.state_dict()}, is_best)