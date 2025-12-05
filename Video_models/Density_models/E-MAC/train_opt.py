import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import math
import os
import sys
from functools import partial
from typing import Dict, Iterable, List, Union, Tuple
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from einops import rearrange
import utils
import utils.data_constants as data_constants
from emac.emac_utils import TransFuse
from emac.criterion import MaskedMSELoss, TVLoss
from emac.input_adapters import PatchedInputAdapter
from emac.output_adapters import SpatialOutputAdapter
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import DICT_MEAN_STD
from utils.density.density_dataset import buildDensityDataset
from utils.optim_factory import LayerDecayValueAssigner, create_optimizer
from utils.task_balancing import NoWeightingStrategy, UncertaintyWeightingStrategy
from utils.pos_embed import interpolate_pos_embed_multimae

DOMAIN_CONF = {"rgb": {"channels": 3, "stride_level": 1, "input_adapter": partial(PatchedInputAdapter, num_channels=3), "loss": MaskedMSELoss},
               "density": {"channels": 1, "stride_level": 1, "input_adapter": partial(PatchedInputAdapter, num_channels=1),
                           "output_adapter": partial(SpatialOutputAdapter, num_channels=1), "loss": MaskedMSELoss}}

def setup_seed(args):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

def get_model(args):
    input_adapters = {domain: DOMAIN_CONF[domain]["input_adapter"](stride_level=DOMAIN_CONF[domain]["stride_level"], patch_size_full=args.patch_size, image_size=args.input_size) for domain in args.in_domains}
    output_adapters = {domain: DOMAIN_CONF[domain]["output_adapter"](stride_level=DOMAIN_CONF[domain]["stride_level"], patch_size_full=args.patch_size, image_size=args.input_size,
                                                                     num_channels=1, dim_tokens_enc=768, dim_tokens=args.decoder_dim, depth=args.decoder_depth, num_heads=args.decoder_num_heads,
                                                                     use_task_queries=args.decoder_use_task_queries, task=domain, context_tasks=list(args.in_domains)) for domain in args.out_domains}
    fuse_module = TransFuse(stride_level=1, patch_size_full=args.patch_size, image_size=args.input_size, num_channels=1, dim_tokens_enc=768, dim_tokens=args.decoder_dim, num_heads=args.decoder_num_heads)
    if args.extra_norm_pix_loss:
        output_adapters["norm_rgb"] = DOMAIN_CONF["rgb"]["output_adapter"](stride_level=DOMAIN_CONF["rgb"]["stride_level"], patch_size_full=args.patch_size)
    model = create_model(args.model, input_adapters=input_adapters, output_adapters=output_adapters, fuse_module=fuse_module, num_global_tokens=args.num_global_tokens, drop_path_rate=args.drop_path)
    return model

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, tasks_loss_fn: Dict[str, torch.nn.Module], loss_balancer: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = None, max_skip_norm: float = None, lr_scheduler=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_encoded_tokens: int = 196, in_domains: List[str] = [], loss_on_unmasked: bool = True, alphas: float = 1.0, sample_tasks_uniformly: bool = False,
                    extra_norm_pix_loss: bool = False, fp32_output_adapters: List[str] = [], print_freq: int = 10, total_num_tokens: int = 400, is_mask_inputs: bool = False,
                    loss_weights: Dict[str, float] = {}, MEAN: float = 0.0, STD: float = 1.0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    tv_loss = TVLoss().cuda()
    fuse_loss = MaskedMSELoss().cuda()
    opt_loss = MaskedMSELoss().cuda()
    for step, x in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        tasks_dict = {task: tensor.to('cuda', non_blocking=True) for task, tensor in x.items() if task != "name"}
        input_dict = {task: tensor for task, tensor in tasks_dict.items() if task in in_domains}
        if "mask" in tasks_dict.keys():
            mask = tasks_dict["mask"]
        B = input_dict["rgb"].shape[0]
        with torch.cuda.amp.autocast(enabled=True):
            if is_mask_inputs:
                task_masks = {}
                task_masks["rgb"] = torch.zeros(B, total_num_tokens).cuda().detach().clone()
                task_masks["density"] = torch.ones(B, total_num_tokens).cuda().detach().clone()
                preds, masks, pred_fuse, img_warp, preds_prev, preds_prev_warp = model(input_dict, num_encoded_tokens=total_num_tokens, mask_inputs=True, task_masks=task_masks)
            else:
                # [6, 1, 512, 512], [6, 1024], [6, 1, 512, 512], [6, 3, 512, 512], [6, 1, 512, 512], [6, 1, 512, 512]
                preds, masks, pred_fuse, img_warp, preds_prev, preds_prev_warp = model(input_dict, num_encoded_tokens=num_encoded_tokens, alphas=alphas,
                                                                                       sample_tasks_uniformly=sample_tasks_uniformly, fp32_output_adapters=fp32_output_adapters)
            if "mask" in tasks_dict.keys():
                preds["density"][mask == 0] = -MEAN / STD
                preds_prev["density"][mask == 0] = -MEAN / STD
                preds_prev_warp[mask == 0] = -MEAN / STD
                pred_fuse[mask == 0] = -MEAN / STD
            if extra_norm_pix_loss:
                tasks_dict["norm_rgb"] = tasks_dict["rgb"]
                masks["norm_rgb"] = masks.get("rgb", None)
            task_losses = {}
            for task in preds:
                target = tasks_dict[task][:, :, -1] # [6, 1, 512, 512]
                img_target = tasks_dict["rgb"][:, :, -1] # [6, 3, 512, 512]
                with torch.cuda.amp.autocast(enabled=False):
                    if loss_on_unmasked:
                        if "cur" in loss_weights:
                            task_losses[task] = loss_weights["cur"] * (tasks_loss_fn[task](preds[task].float(), target.float()))
                        if "opt" in loss_weights:
                            task_losses["opt"] = loss_weights["opt"] * opt_loss(img_warp.float(), img_target.float())
                        task_losses["fuse"] = loss_weights["fus"] * fuse_loss(pred_fuse.float(), target.float())
                    else:
                        task_losses[task] = 10 * tasks_loss_fn[task](preds[task].float(), target.float(), mask=masks.get(task, None))
                    if "tv" in loss_weights:
                        task_losses["tv"] = loss_weights["tv"] * (tv_loss(pred_fuse.float()).mean())
            with torch.cuda.amp.autocast(enabled=False):
                weighted_task_losses = loss_balancer(task_losses)
                loss = sum(weighted_task_losses.values())
        with torch.cuda.amp.autocast(enabled=False):
            loss_value = sum(task_losses.values()).item()
            task_loss_values = {f"{task}_loss": l.item() for task, l in task_losses.items()}
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        with torch.cuda.amp.autocast(enabled=False):
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm, parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {"[Epoch] " + k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, epoch: int, in_domains: List[str] = [], mode: str = "val", print_freq: int = 50, total_num_tokens: int = 400,
             MEAN: float = 0.0, STD: float = 1.0, args=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(f"{mode}_err", utils.SmoothedValue(window_size=1, fmt="{value:.4f} ({global_avg:.4f})"))
    metric_logger.add_meter(f"{mode}_gt", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter(f"{mode}_pred", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    if mode == "val":
        header = "(Eval) Epoch: [{}]".format(epoch)
    elif mode == "test":
        header = "(Test) Epoch: [{}]".format(epoch)
    else:
        raise ValueError(f"Invalid eval mode {mode}")
    mae = 0.0
    mse = 0.0
    for x in metric_logger.log_every(data_loader, print_freq, header):
        tasks_dict = {task: tensor.to('cuda', non_blocking=True) for task, tensor in x.items() if task != "name"}
        input_dict = {task: tensor for task, tensor in tasks_dict.items() if task in in_domains}
        B = input_dict["rgb"].shape[0]
        task_masks = {}
        task_masks["rgb"] = torch.zeros(B, total_num_tokens).cuda() # [1, 1024]
        task_masks["density"] = torch.ones(B, total_num_tokens).cuda() # [1, 1024]
        if "mask" in tasks_dict.keys():
            mask = tasks_dict["mask"]
        with torch.cuda.amp.autocast():
            target = tasks_dict["density"][:, :, -1] # [1, 1, 512, 512]
            input_dict["density"] = torch.rand_like(input_dict["density"]).cuda() # [1, 1, 2, 512, 512]
            preds, masks, pred_fuse, img_warp, preds_prev, preds_prev_warp = model(input_dict, num_encoded_tokens=total_num_tokens, mask_inputs=True, task_masks=task_masks)
            if "mask" in tasks_dict.keys():
                preds["density"][mask == 0] = -MEAN / STD
                preds_prev["density"][mask == 0] = -MEAN / STD
                preds_prev_warp[mask == 0] = -MEAN / STD
                pred_fuse[mask == 0] = -MEAN / STD
            input_dict["density"] = tasks_dict["density"]
        pred_count = (pred_fuse * STD + MEAN).sum().cpu().item()
        gt_count = (target * STD + MEAN).sum().cpu().item()
        err = abs(pred_count - gt_count)
        mae += err
        mse += err**2
        metric_logger.update(**{f"{mode}_err": err})
        metric_logger.update(**{f"{mode}_gt": gt_count})
        metric_logger.update(**{f"{mode}_pred": pred_count})
    mae = mae / len(data_loader)
    mse = (mse / len(data_loader)) ** 0.5
    print('Epoch: [{}/{}], MAE: {:.2f}, RMSE: {:.2f}'.format(epoch + 1, args.epochs, mae, mse))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, mae, mse

def main(args):
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)
    utils.init_distributed_mode(args)
    setup_seed(args)
    args.in_domains = args.in_domains.split("-")
    args.out_domains = args.out_domains.split("-")
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))
    # model
    model = get_model(args)
    # loss
    if args.task_balancer == "uncertainty":
        loss_balancer = UncertaintyWeightingStrategy(tasks=["density", "count"])
    else:
        loss_balancer = NoWeightingStrategy()
    tasks_loss_fn = {domain: DOMAIN_CONF[domain]["loss"](patch_size=args.patch_size, stride=DOMAIN_CONF[domain]["stride_level"]) for domain in args.out_domains}
    if args.extra_norm_pix_loss:
        tasks_loss_fn["norm_rgb"] = DOMAIN_CONF["rgb"]["loss"](patch_size=args.patch_size, stride=DOMAIN_CONF["rgb"]["stride_level"], norm_pix=True)
    # train and val loader
    dataset_train = buildDensityDataset(data_root=args.data_path, split="train", image_size=args.input_size, max_images=args.max_train_images, dataset_name=args.dataset,
                                        clip_size=args.data_clip_size, stride=args.data_stride, MEAN=[args.DATA_MEAN], STD=[args.DATA_STD])
    dataset_val = buildDensityDataset(data_root=args.data_path, split="val", image_size=args.input_size, max_images=args.max_val_images, dataset_name=args.dataset,
                                      clip_size=args.data_clip_size, stride=args.data_stride, MEAN=[args.DATA_MEAN], STD=[args.DATA_STD])
    if True: # DDP training
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = (len(dataset_train) // args.batch_size // num_tasks)
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True, drop_last=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    # resume training
    if args.ckpt_multi is not None:
        checkpoint = torch.load(args.ckpt_multi, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        for k in list(checkpoint_model.keys()):
            if "semseg" in k:
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if "output_adapters" in k and "output_adapters.rgb" not in k and "output_adapters.density" not in k:
                del checkpoint_model[k]
        copy_layers = []
        interpolate_pos_embed_multimae(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # print(msg)
        model.state_dict()["input_adapters.density.proj.weight"].copy_(checkpoint_model["input_adapters.rgb.proj.weight"].sum(1, keepdim=True))
        model.state_dict()["input_adapters.density.pos_emb"].copy_(checkpoint_model["input_adapters.rgb.pos_emb"])
        model.state_dict()["output_adapters.density.pos_emb"].copy_(rearrange(checkpoint_model["input_adapters.rgb.pos_emb"], "b (d c) h w  -> b d c h w", c=3).sum(2))
        model.state_dict()["output_adapters.density.out_proj.weight"].copy_(checkpoint_model["output_adapters.rgb.out_proj.weight"].reshape(256, 3, -1).sum(1, keepdim=True).reshape(256, -1))
        copy_layers.append("input_adapters.density.pos_emb")
        for k in list(model.state_dict().keys()):
            if "output_adapters.density" in k and "out_proj" not in k and "pos_emb" not in k and "addconv" not in k and "smoothconv" not in k:
                # print(k)
                model.state_dict()[k].copy_(checkpoint_model[k.replace("density", "rgb")])
                copy_layers.append(k)
        model.fuse.state_dict()["out_proj.weight"].copy_(checkpoint_model["output_adapters.rgb.out_proj.weight"].reshape(256, 3, -1).sum(1, keepdim=True).reshape(256, -1))
        model.fuse.state_dict()["pos_emb"].copy_(rearrange(checkpoint_model["input_adapters.rgb.pos_emb"], "b (d c) h w  -> b d c h w", c=3).sum(2))
        for k in list(model.fuse.state_dict().keys()):
            if "out_proj" not in k and "pos_emb" not in k:
                model.fuse.state_dict()[k].copy_(checkpoint_model["output_adapters.rgb." + k])
        # print(copy_layers)
        model.pwc.load_state_dict(torch.load("cfgs/pwc_net.pth.tar"))
        print('Load ckpt of MultiMAE from:', args.ckpt_multi)
    model.cuda()
    loss_balancer.cuda()
    model_without_ddp = model
    loss_balancer_without_ddp = loss_balancer
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters / 1e6} M")
    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256
    skip_weight_decay_list = model.no_weight_decay()
    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0: # idx = 0: input adapters, idx > 0: transformer layers
        layer_decay_values = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(layer_decay_values)
    else:
        assigner = None
    if assigner is not None:
        print("Assigned values: %s" % str(assigner.values))
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    if args.distributed and args.task_balancer != "none":
        loss_balancer = torch.nn.parallel.DistributedDataParallel(loss_balancer, device_ids=[args.gpu])
        loss_balancer_without_ddp = loss_balancer.module
    # optimizer
    optimizer = create_optimizer(args, model_without_ddp, skip_list=skip_weight_decay_list, get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                 get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()
    lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    min_val_loss = np.inf
    best_epoch = 0
    if args.eval_first:
        val_stats, mae, mse = evaluate(model=model, data_loader=data_loader_val, epoch=0, in_domains=args.in_domains, mode="val", print_freq=args.val_print_freq,
                                       total_num_tokens=args.total_num_tokens, MEAN=args.DATA_MEAN, STD=args.DATA_STD, args=args)
        print(val_stats, val_stats.keys())
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model=model, data_loader=data_loader_train, tasks_loss_fn=tasks_loss_fn, loss_balancer=loss_balancer, optimizer=optimizer, epoch=epoch,
                                      loss_scaler=loss_scaler, max_norm=args.clip_grad, max_skip_norm=args.skip_grad, start_steps=epoch * num_training_steps_per_epoch,
                                      lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, num_encoded_tokens=args.num_encoded_tokens,
                                      in_domains=args.in_domains, loss_on_unmasked=args.loss_on_unmasked, alphas=args.alphas, sample_tasks_uniformly=args.sample_tasks_uniformly,
                                      extra_norm_pix_loss=args.extra_norm_pix_loss, fp32_output_adapters=args.fp32_output_adapters.split("-"),
                                      print_freq=args.train_print_freq, total_num_tokens=args.total_num_tokens, is_mask_inputs=args.is_mask_inputs,
                                      loss_weights=args.loss_weights, MEAN=args.DATA_MEAN, STD=args.DATA_STD)
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, loss_balancer=loss_balancer_without_ddp, epoch=epoch)
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stats, mae, mse = evaluate(model=model, data_loader=data_loader_val, epoch=epoch, in_domains=args.in_domains, mode="val", print_freq=args.val_print_freq,
                                           total_num_tokens=args.total_num_tokens, MEAN=args.DATA_MEAN, STD=args.DATA_STD, args=args)
            print("Epoch: [{}/{}], MAE: {:.2f}, MSE: {:.2f}, Min MAE: {:.2f}".format(epoch + 1, args.epochs, mae, mse, min_val_loss))
            # save best model
            if mae < min_val_loss:
                min_val_loss = mae
                best_epoch = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch="best")
                print(f"New best val loss: {min_val_loss:.2f}")
            else:
                print(f"Current best val loss: {min_val_loss:.2f} at epoch: {best_epoch + 1}")
            log_stats = {**{f"train/{k}": v for k, v in train_stats.items()}, **{f"val/{k}": v for k, v in val_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}
        else:
            log_stats = {**{f"train/{k}": v for k, v in train_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == "__main__":
    config_parser = parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfgs/density/DroneBird.yaml", type=str)
    # general config
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='DroneBird')
    parser.add_argument("--output_dir", default="saved_dronebird", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false")
    parser.add_argument("--ckpt_multi", default=None, type=str)
    parser.set_defaults(auto_resume=True)
    # training config
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=1600, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--save_ckpt", action="store_true")
    parser.set_defaults(save_ckpt=True)
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--eval_first", default=False, action="store_true") # eval model before training
    parser.add_argument("--max_train_images", default=1000, type=int)
    parser.add_argument("--max_val_images", default=100, type=int)
    parser.add_argument("--max_test_images", default=54514, type=int)
    parser.add_argument("--in_domains", default="rgb-density-rgb", type=str)
    parser.add_argument("--out_domains", default="rgb-density-rgb", type=str)
    parser.add_argument("--standardize_depth", action="store_true")
    parser.add_argument("--no_standardize_depth", action="store_false")
    parser.set_defaults(standardize_depth=False)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--no_find_unused_params", action="store_false")
    parser.set_defaults(find_unused_params=True)
    parser.add_argument("--train_print_freq", default=10, type=int)
    parser.add_argument("--val_print_freq", default=50, type=int)
    # loss config
    parser.add_argument("--extra_norm_pix_loss", action="store_true")
    parser.add_argument("--extra_unnorm_den_loss", action="store_true")
    parser.add_argument("--no_extra_norm_pix_loss", action="store_false")
    parser.set_defaults(extra_norm_pix_loss=True)
    parser.add_argument("--use_opt_loss", default=True, action="store_true")
    parser.add_argument("--use_cur_loss", default=True, action="store_true")
    parser.add_argument("--use_tv_loss", default=True, action="store_true")
    parser.add_argument("--loss_weights", default={"opt": 1.0, "cur": 10.0, "tv": 20.0, "fus": 10.0})
    # model config
    parser.add_argument("--model", default="pretrain_multimae_base", type=str)
    parser.add_argument("--is_mask_inputs", default=False, type=bool)
    parser.add_argument("--total_num_tokens", default=400, type=int)
    parser.add_argument("--num_encoded_tokens", default=98, type=int)
    parser.add_argument("--num_global_tokens", default=1, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--input_size", default=224, type=Union[int, Tuple[int]])
    parser.add_argument("--alphas", type=float, default=1.0)
    parser.add_argument("--sample_tasks_uniformly", default=False, action="store_true")
    parser.add_argument("--decoder_use_task_queries", default=True, action="store_true")
    parser.add_argument("--decoder_use_xattn", default=True, action="store_true")
    parser.add_argument("--decoder_dim", default=256, type=int)
    parser.add_argument("--decoder_depth", default=2, type=int)
    parser.add_argument("--decoder_num_heads", default=8, type=int)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--loss_on_unmasked", default=False, action="store_true")
    parser.add_argument("--no_loss_on_unmasked", action="store_false")
    parser.set_defaults(loss_on_unmasked=True)
    # optimizer config
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=[0.9, 0.95], type=float, nargs="+")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--skip_grad", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--weight_decay_end", type=float, default=None)
    parser.add_argument("--decoder_decay", type=float, default=None)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--blr", type=float, default=1e-4)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--task_balancer", type=str, default="none")
    parser.add_argument("--balancer_lr_scale", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--fp32_output_adapters", type=str, default="")
    # augmentation config
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--train_interpolation", type=str, default="bicubic")
    parser.add_argument("--cilp_size", type=int, default=5)
    # dataset config
    parser.add_argument("--dataset", default="DroneBird", type=str)
    parser.add_argument("--data_path", default=data_constants.IMAGENET_TRAIN_PATH, type=str)
    parser.add_argument("--imagenet_default_mean_and_std", default=True, action="store_true")
    parser.add_argument("--data_clip_size", default=2, type=int)
    parser.add_argument("--data_stride", default=1, type=int)
    parser.add_argument("--DATA_MEAN", default=None, type=float)
    parser.add_argument("--DATA_STD", default=None, type=float)
    # DDP config
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    # load config
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    if args.DATA_MEAN is None or args.DATA_STD is None:
        args.DATA_MEAN, args.DATA_STD = DICT_MEAN_STD[args.dataset]["MEAN"], DICT_MEAN_STD[args.dataset]["STD"]
    os.makedirs(args.output_dir, exist_ok=True)
    print('Training dataset:', args.dataset)
    main(args)