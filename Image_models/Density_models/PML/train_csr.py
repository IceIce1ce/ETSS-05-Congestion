import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import os
import torch
from easydict import EasyDict as edict
from termcolor import cprint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from datasets import build_dataset
from eingine.trainer import evaluate_counting, train_one_epoch
from misc import utils
from misc.saver_builder import Saver
from misc.utils import MetricLogger,is_main_process
from models.CSRNet import CSRNet
from models.loss import build_loss
from optimizer import loss_weight_builder, optimizer_builder, scheduler_builder

def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        k = k[11:]
        state_dict[k] = v
    return state_dict

def main(args):
    utils.init_distributed_mode(args)
    utils.set_randomseed(42 + utils.get_rank())
    # model
    model = model_without_ddp = CSRNet()
    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    # train and val loader
    dataset_train = build_dataset(image_set='train', args=args.Dataset.train)
    dataset_val = build_dataset(image_set='val', args=args.Dataset.val)
    sampler_train = DistributedSampler(dataset_train) if args.distributed else None
    sampler_val = DistributedSampler(dataset_val, shuffle=False) if args.distributed else None
    loader_train = DataLoader(dataset_train, batch_size=args.Dataset.train.batch_size, sampler=sampler_train, shuffle=(sampler_train is None), num_workers=0, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=args.Dataset.val.batch_size, sampler=sampler_val, shuffle=False, num_workers=0, pin_memory=True)
    # optimizer
    optimizer = optimizer_builder(args.Optimizer, model_without_ddp)
    scheduler = scheduler_builder(args.Scheduler, optimizer)
    if args.Scheduler.ema:
        def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
            return args.Scheduler.ema_weight * averaged_model_parameter + (1 - args.Scheduler.ema_weight) * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
        ema_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy=args.Scheduler.ema_annel_strategy, anneal_epochs=args.Scheduler.ema_annel_epochs, swa_lr=args.Scheduler.ema_lr)
        ema_saver = Saver(args.Saver, is_ema=True)
    else:
        ema_model = None
        ema_scheduler = None
    # loss
    loss_weight = loss_weight_builder(args.Loss_Weight)
    counting_criterion = build_loss(args.Loss.counting)
    saver = Saver(args.Saver)
    if args.Misc.use_tensorboard:
        tensorboard_writer = SummaryWriter(args.Misc.tensorboard_dir)
    for epoch in range(args.Misc.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_logger = MetricLogger(args.Logger)
        val_logger=MetricLogger(args.Logger)
        stats = edict()
        stats.train_stats = train_one_epoch(model, counting_criterion, loader_train, optimizer, train_logger, loss_weight, epoch, args)
        if args.Scheduler.ema and epoch > args.Scheduler.ema_start_epoch:
            ema_model.update_parameters(model)
            ema_scheduler.step()
            torch.optim.swa_utils.update_bn(loader_train, ema_model)
            stats.ema_test_stats = evaluate_counting(ema_model, counting_criterion, loader_val, val_logger, args)
            ema_saver.save_on_master(ema_model, optimizer, scheduler, epoch, stats)
        else:
            scheduler.step()
            stats.ema_test_stats = {}
            stats.ema_test_stats = {}
        stats.test_stats = evaluate_counting(model, counting_criterion, loader_val, val_logger, args)
        saver.save_on_master(model, optimizer, scheduler, epoch, stats)
        log_stats = {**{f'train_{k}': v for k, v in stats.train_stats.items()}, **{f'val_{k}': v for k, v in stats.test_stats.items()},
                     **{f'ema_val_{k}': v for k, v in stats.ema_test_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            for key, value in log_stats.items():
                cprint(f'{key}:{value}', 'green')
                if args.Misc.use_tensorboard:
                    tensorboard_writer.add_scalar(key, value, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument("--config", default="configs/sha.json")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
        cfg = edict(configs)
    if is_main_process():
        if not os.path.exists(cfg.Saver.save_dir):
            os.makedirs(cfg.Saver.save_dir)
    main(cfg)