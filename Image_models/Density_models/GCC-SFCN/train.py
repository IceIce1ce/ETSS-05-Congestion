import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from models.CC import CrowdCounter
from loading_data import loading_data
from config import cfg
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

exp_name = cfg.TRAIN.EXP_NAME
if not os.path.exists(cfg.TRAIN.EXP_PATH):
    os.makedirs(cfg.TRAIN.EXP_PATH)
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)
log_txt = cfg.TRAIN.EXP_PATH + '/' + exp_name + '/' + exp_name + '.txt'
pil_to_tensor = standard_transforms.ToTensor()
train_record = {'best_mae': 1e20, 'mse':1e20,'corr_loss': 0, 'corr_epoch': -1, 'best_model_name': ''}
train_set, train_loader, val_set, val_loader, restore_transform = loading_data()
rand_seed = cfg.TRAIN.SEED
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def main():
    # config
    cfg_file = open('config.py',"r")
    cfg_lines = cfg_file.readlines()
    # log
    with open(log_txt, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True
    # model
    net = CrowdCounter().cuda()
    if cfg.TRAIN.PRE_GCC:
        net.load_state_dict(torch.load(cfg.TRAIN.PRE_GCC_MODEL))
    net.train()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    i_tb = 0
    # train
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        if epoch > cfg.TRAIN.LR_DECAY_START:
            scheduler.step()
        i_tb = train(train_loader, net, optimizer, epoch, i_tb)
        # test
        # if epoch % cfg.VAL.FREQ == 0 or epoch > cfg.VAL.DENSE_START:
        if epoch % 1 == 0:
            validate(val_loader, net, epoch, restore_transform)

def train(train_loader, net, optimizer, epoch, i_tb):
    for i, data in enumerate(train_loader, 0):
        img, gt_map, gt_cnt = data # [4, 3, 576, 768], [4, 576, 768], [4]
        img = Variable(img).cuda()
        gt_map = Variable(gt_map).cuda()
        optimizer.zero_grad()
        pred_map = net(img, gt_map) # [4, 576, 768]
        loss = net.loss
        loss.backward()
        optimizer.step()
        pred_map = pred_map / 100.
        if (i + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            i_tb = i_tb + 1
            writer.add_scalar('train_loss', loss.data.item(), i_tb)
            print('Epoch: {}, Iter: [{}/{}], Loss: {:.4f}'.format(epoch + 1, i + 1, len(train_loader), loss.data.item()))
            print('GT: {:.4f}, Pred: {:.4f}'.format(gt_cnt[0], pred_map[0, :, :].sum().item()))
    return i_tb

def validate(val_loader, net, epoch, restore):
    net.eval()
    val_loss = []
    mae = 0.0
    mse = 0.0
    for vi, data in enumerate(val_loader, 0):
        img, gt_map, gt_count = data # [2, 3, 768, 1024], [2, 768, 1024], [2]
        img = Variable(img, volatile=True).cuda()
        gt_map = Variable(gt_map, volatile=True).cuda()
        gt_count = gt_count.numpy()
        pred_map = net(img, gt_map) # [2, 768, 1024]
        val_loss.append(net.loss.item())
        pred_map = pred_map / 100.
        pred_map = pred_map.data.cpu().numpy()
        gt_map = gt_map / 100.
        gt_map = gt_map.data.cpu().numpy()
        for i_img in range(pred_map.shape[0]):
            pred_cnt_tmp = np.sum(pred_map[i_img])
            gt_count_tmp = gt_count[i_img]
            mae += abs(gt_count_tmp - pred_cnt_tmp)
            mse += ((gt_count_tmp - pred_cnt_tmp) * (gt_count_tmp - pred_cnt_tmp))
        x = []
        if vi == 0:
            for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map)):
                if idx > cfg.VIS.VISIBLE_NUM_IMGS:
                    break
                pil_input = restore(tensor[0])
                pil_output = torch.from_numpy(tensor[1] / (tensor[1].max() + 1e-10)).repeat(3, 1, 1) # [3, 768, 1024]
                pil_label = torch.from_numpy(tensor[2] / (tensor[2].max() + 1e-10)).repeat(3, 1, 1) # [3, 768, 1024]
                x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output])
            x = torch.stack(x, 0) # [6, 3, 768, 1024]
            x = vutils.make_grid(x, nrow=3, padding=5) # [3, 1551, 3092]
            writer.add_image(exp_name + '_epoch_' + str(epoch+1), (x.numpy() * 255).astype(np.uint8))
    mae = mae / val_set.get_num_samples()
    mse = np.sqrt(mse / val_set.get_num_samples())
    loss = np.mean(np.array(val_loss))
    writer.add_scalar('val_loss', loss, epoch + 1)
    writer.add_scalar('mae', mae, epoch + 1)
    writer.add_scalar('mse', mse, epoch + 1)
    snapshot_name = 'ep_%d_mae_%.1f_mse_%.1f' % (epoch + 1, mae, mse)
    if mae < train_record['best_mae']:
        train_record['best_mae'] = mae
        train_record['mse'] = mse
        train_record['corr_epoch'] = epoch + 1
        train_record['corr_loss'] = loss        
        train_record['best_model_name'] = snapshot_name
        with open(log_txt, 'a') as f:
            f.write(snapshot_name + '\n')
        to_saved_weight = net.state_dict()
        torch.save(to_saved_weight, os.path.join(cfg.TRAIN.EXP_PATH, exp_name, snapshot_name + '.pth'))
    print('MAE: {:.4f}, MSE: {:.4f}, Testing loss: {:.4f}'.format(mae, mse, loss))
    print('Best MAE: {:.4f}, Best MSE: {:.4f}, Loss: {:.4f} at epoch: {}'.format(train_record['best_mae'], train_record['mse'], train_record['corr_loss'], train_record['corr_epoch']))
    net.train()

if __name__ == '__main__':
    main()
