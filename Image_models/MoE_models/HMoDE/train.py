import numpy as np
from torch import optim
from torch.autograd import Variable
from HMoDE import HMoDE
import torch.nn as nn
import torch
import os
from datasets.SHHA.loading_data import loading_data
from datasets.SHHA.setting import cfg_data
import argparse
import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def train(train_loader, net, optimizer, epoch):
    net.train()
    # loss
    mseloss = nn.MSELoss(reduction='sum').cuda()
    for i, data in enumerate(train_loader, 0):
        img, gt_map = data # [1, 3, 359, 329], [1, 359, 329]
        img = Variable(img).cuda()
        gt_map = Variable(gt_map).cuda()
        amp_gt = (gt_map > (1e-5 * cfg_data.LOG_PARA)).float().unsqueeze(1) # [1, 1, 359, 329]
        pred_maps, amp, imp_loss = net(img) # [1, 1, 359, 329] * len(7), [1, 1, 22, 20], [1]
        optimizer.zero_grad()
        loss = 0.
        rel_loss = 0.
        for j in range(len(pred_maps)):
            loss += (2**(int(j / 3))) * mseloss(pred_maps[j], gt_map)
        amp = nn.functional.interpolate(amp, amp_gt.shape[2:], mode='nearest')
        cross_entropy_loss = (amp_gt * torch.log(amp+1e-10) + (1 - amp_gt) * torch.log(1 - amp+1e-10)) * -1
        loss = loss + rel_loss + torch.sum(imp_loss) + torch.sum(cross_entropy_loss)
        loss = loss / pred_maps[0].shape[0]
        loss.backward()
        optimizer.step()
        if (i + 1) % cfg_data.PRINT_FREQ == 0:
            loss = mseloss(pred_maps[0].squeeze(), gt_map)
            print('Epoch: {}, Iter: [{}/{}], Loss: {:.4f}'.format(epoch + 1, i + 1, len(train_loader), torch.sum(loss).item()))
    if len(cfg_data.GPU_ID) > 1:
        to_saved_weight = net.module.state_dict()
    else:
        to_saved_weight = net.state_dict()
    state = {'epoch': epoch, 'model': to_saved_weight, 'optimizer': optimizer.state_dict(), 'record': train_record}
    model_path = os.path.join(cfg_data.EXP_PATH, 'latestmodel.pth')
    torch.save(state, model_path)
    return model_path

def validate(val_loader, val_set, epoch):
    torch.cuda.empty_cache()
    mseloss = nn.MSELoss(reduction='sum').cuda()
    net = HMoDE(False)
    net.load_state_dict(torch.load(os.path.join(cfg_data.EXP_PATH, 'latestmodel.pth'))['model'])
    net.cuda()
    net.eval()
    # val_loss = []
    mae = 0.0
    mse = 0.0
    for vi, data in enumerate(val_loader, 0):
        img, gt_map = data
        with torch.no_grad():
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            # pred_map = net(img)[0]
            pred_map = net(img)[0][-1]
            # loss = mseloss(pred_map, gt_map)
            # val_loss.append(loss.item())
            pred_map = pred_map.data.cpu().numpy() / cfg_data.LOG_PARA
            gt_map = gt_map.data.cpu().numpy() / cfg_data.LOG_PARA
            gt_count = np.sum(gt_map)
            pred_cnt = np.sum(pred_map)
            mae += abs(gt_count - pred_cnt)
            mse += ((gt_count - pred_cnt) * (gt_count - pred_cnt))
    mae = mae / val_set.get_num_samples()
    mse = np.sqrt(mse / val_set.get_num_samples())
    # loss = np.mean(val_loss)
    if mae < train_record['best_mae']:
        train_record['best_mae'] = mae
        train_record['mse'] = mse
        train_record['corr_epoch'] = epoch + 1
        # train_record['corr_loss'] = loss
        to_saved_weight = net.state_dict()
        state = {'model': to_saved_weight}
        model_path = os.path.join(cfg_data.EXP_PATH, 'best_model.pth')
        torch.save(state, model_path)
    # print('MAE: {:.4f}, MSE: {:.4f}, Testing loss: {:.4f}'.format(mae, mse, loss))
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))
    # print('Best MAE: {:.4f}, Best MSE: {:.4f} Testing loss: {:.4f} at epoch: {}'.format(train_record['best_mae'], train_record['mse'], train_record['corr_loss'], train_record['corr_epoch']))
    print('Best MAE: {:.4f}, Best MSE: {:.4f} at epoch: {}'.format(train_record['best_mae'], train_record['mse'], train_record['corr_epoch']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A_final')
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    args = parser.parse_args()

    print('Training dataset:', args.input_dir.split('/')[1])
    cfg_data.DATA_PATH = args.input_dir
    cfg_data.EXP_PATH = args.output_dir
    log_txt = cfg_data.EXP_PATH + '/' + cfg_data.EXP_NAME + '.txt'
    if not os.path.exists(cfg_data.EXP_PATH):
        os.makedirs(cfg_data.EXP_PATH)
    train_record = {'best_mae': 1e20, 'mse': 1e20, 'corr_loss': 0, 'corr_epoch': -1, 'best_model_name': ''}
    setup_seed(cfg_data.SEED)
    # train and test loader
    train_set, train_loader, val_set, val_loader = loading_data()
    load = False
    begin = 0
    # model
    net = HMoDE(True)
    # net = nn.DataParallel(net)
    net = net.cuda()
    net.train()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=2e-5)
    stepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    # resume training
    if load:
        checkpoint = torch.load(os.path.join(cfg_data.EXP_PATH, 'latestmodel.pth'))
        net.load_state_dict(checkpoint['model'])
        begin = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_record['best_mae'] = checkpoint['record']['best_mae']
        train_record['mse'] = checkpoint['record']['mse']
        train_record['corr_epoch'] = checkpoint['record']['corr_epoch']
        train_record['corr_loss'] = checkpoint['record']['corr_loss']
        print('Load ckpt from: {}', os.path.join(cfg_data.EXP_PATH, 'latestmodel.pth'))
    for epoch in range(begin, cfg_data.MAX_EPOCH):
        model_path = train(train_loader, net, optimizer, epoch)
        if epoch + 1 >= 100:
            validate(val_loader, val_set, epoch)
        if (epoch + 1) == 100:
            stepLR.step()
