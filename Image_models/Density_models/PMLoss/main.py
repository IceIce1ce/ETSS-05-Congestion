import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import torch
from torch import optim
from timm.utils import AverageMeter
from config import get_config
from models import build_model
from datasets import build_loader
from losses import build_loss
from lr_scheduler import build_scheduler
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, CurvePlotter, setup_seed

def train_one_epoch(config, model, criterion, data_loader, optimizer, lr_scheduler, epoch):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    print_freq = num_steps // max(config.PRINT_FREQ - 1, 1)
    for idx, (samples, dotseq, imgid) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        dotseq = [d.cuda(non_blocking=True) for d in dotseq]
        denmap = model(samples)
        loss = criterion(denmap, dotseq, samples.size(-1) // denmap.size(-1))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        torch.cuda.synchronize()
        if lr_scheduler is not None:
            lr_scheduler.step()
        loss_meter.update(loss.item(), samples.size(0))
        norm_meter.update(grad_norm)
        if idx % print_freq == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            print('[Training]: Epoch: [{}/{}], Loss: {:.4f} ({:.4f})'.format(epoch + 1, config.TRAIN.EPOCHS, loss_meter.val, loss_meter.avg))

@torch.inference_mode()
def validate(config, data_loader, model, epoch):
    model.eval()
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()
    cnts = []
    num_steps = len(data_loader)
    print_freq = num_steps // max(config.PRINT_FREQ - 1, 1)
    for idx, (images, dotseq, imgid) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        dotseq = [d.cuda(non_blocking=True) for d in dotseq]
        cnt = torch.tensor([d.size(0) for d in dotseq]).float().cuda()
        bsize = images.size(0)
        output = model(images)
        outnum = output.sum(dim=(1, 2, 3)) / config.MODEL.FACTOR
        diff = torch.abs(outnum - cnt)
        cnts.append((outnum.item(), cnt.item()))
        mae, mse = diff.mean().item(), (diff ** 2).mean().item()
        mae_meter.update(mae, bsize)
        mse_meter.update(mse, bsize)
        if idx % print_freq == 0 or idx == num_steps - 1:
            print('[Testing]: Epoch: [{}/{}], MAE: {:.2f} ({:.2f}), MSE: {:.2f} ({:.2f})'.format(epoch + 1, config.TRAIN.EPOCHS, mae_meter.val, mae_meter.avg, mse_meter.val ** 0.5, mse_meter.avg ** 0.5))
    print('[Testing]: Epoch: [{}/{}], MAE: {:.2f}, MSE: {:.2f}'.format(epoch + 1, config.TRAIN.EPOCHS, mae_meter.avg, mse_meter.avg ** 0.5))
    return mae_meter.avg, mse_meter.avg ** 0.5, cnts

def main(config):
    # train and test loader
    data_loader_train = build_loader(config.DATA, mode='train')
    data_loader_test = build_loader(config.DATA, mode='test')
    # model
    model = build_model(config.MODEL)
    model.cuda()
    model_without_ddp = model
    # loss
    criterion, test_criterion = build_loss(config.MODEL)
    criterion.cuda()
    test_criterion.cuda()
    # optimizer
    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if "encoders" not in n and p.requires_grad]},
                   {"params": [p for n, p in model_without_ddp.named_parameters() if "encoders" in n and p.requires_grad], "lr": config.TRAIN.BACKBONE_LR}]
    optimizer = optim.Adam(param_dicts, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    lr_scheduler = build_scheduler(optimizer, config.TRAIN, len(data_loader_train))
    max_accuracy_test = [1e6] * 2
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT_DIR)
        if resume_file:
            if config.MODEL.RESUME:
                print(f"Changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            print('Resume training from:', resume_file)
        else:
            print('No ckpt found at:', config.OUTPUT_DIR)
    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp)
        if config.EVAL_MODE:
            return
    mae_curve = CurvePlotter("mae", config.OUTPUT_DIR)
    mse_curve = CurvePlotter("mse", config.OUTPUT_DIR)
    test_feq, last_test = 0, -1
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS + 1):
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, lr_scheduler, epoch)
        if epoch - last_test >= test_feq or epoch == (config.TRAIN.EPOCHS):
            mae, mse, _ = validate(config, data_loader_test, model, epoch)
            if test_feq > 0:
                mae_curve.add_and_plot(epoch, mae)
                mse_curve.add_and_plot(epoch, mse)
            if mae * 4 + mse < max_accuracy_test[0] * 4 + max_accuracy_test[1]:
                max_accuracy_test = (mae, mse)
                save_checkpoint(config, "best", model_without_ddp, max_accuracy_test)
                if test_feq > 0:
                    mae_curve.set_key_epoch(epoch)
                    mse_curve.set_key_epoch(epoch)
            print('[Testing]: Epoch: [{}/{}], MAE: {:.2f}, MSE: {:.2f}'.format(epoch + 1, config.TRAIN.EPOCHS, max_accuracy_test[0], max_accuracy_test[1]))
            last_test = epoch
            test_feq = max(int(config.MAX_SAVE_FREQ * (config.SAVE_FREQ_FACTOR ** (epoch / config.TRAIN.EPOCHS))), config.MIN_SAVE_FREQ)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='data/ShanghaiTech/part_A')
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    # training config
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--resume', type=str, default='')
    # testing config
    parser.add_argument('--eval', action='store_true')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    print('Training dataset:', args.type_dataset)
    setup_seed(config.SEED)
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    main(config)