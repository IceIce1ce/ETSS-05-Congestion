import warnings
warnings.filterwarnings('ignore')
from utils.density.density_dataset import buildDensityDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.data_constants import FDST_MEAN, FDST_STD, Mall_MEAN, Mall_STD, DB_MEAN, DB_STD, VSC_MEAN, VSC_STD
from emac.emac import emac_base as e_mac
from emac.output_adapters import SpatialOutputAdapter
from emac.input_adapters import PatchedInputAdapter
from einops import rearrange
import torchvision.transforms.functional as TF
import torch
import numpy as np
from functools import partial
import os
from utils.pos_embed import interpolate_pos_embed_multimae
from emac.emac_utils import TransFuse
import argparse
torch.set_grad_enabled(False)

Normalization = {"FDST": {"MEAN": FDST_MEAN, "STD": FDST_STD}, "Mall": {"MEAN": Mall_MEAN, "STD": Mall_STD},
                 "DroneBird": {"MEAN": DB_MEAN, "STD": DB_STD}, "VSCrowd": {"MEAN": VSC_MEAN, "STD": VSC_STD}}

def make_img_crop(img, img_height, img_width):
    if img.ndim == 3:
        c, h, w = img.shape
    elif img.ndim == 4:
        c, t, h, w = img.shape
    if h < img_height or w < img_width:
        raise ValueError("Image is too small for the desired crop size")
    hc, wc = int(h // img_height), int(w // img_width)

    if img.ndim == 3:
        img = TF.resize(img, (hc * img_height, wc * img_width))
        img = rearrange(img, "c (hc h) (wc w) -> (hc wc) c h w", hc=hc, wc=wc)
    elif img.ndim == 4:
        imgs = torch.zeros((hc * wc, c, t, img_height, img_width))
        for i in range(t):
            img_t = img[:, i]
            img_t = TF.resize(img_t, (hc * img_height, wc * img_width))
            img_t = rearrange(img_t, "c (hc h) (wc w) -> (hc wc) c h w", hc=hc, wc=wc)
            imgs[:, :, i] = img_t
        img = imgs
    return img, hc, wc

def main(args):
    DOMAIN_CONF = {"rgb": {"channels": 3, "stride_level": 1, "input_adapter": partial(PatchedInputAdapter, num_channels=3, stride_level=1), "output_adapter": partial(SpatialOutputAdapter, num_channels=3)},
                   "density": {"channels": 1, "stride_level": 1, "input_adapter": partial(PatchedInputAdapter, num_channels=1, stride_level=1), "output_adapter": partial(SpatialOutputAdapter, num_channels=1)}}
    DOMAINS = ["rgb", "density"]
    input_adapters = {domain: DOMAIN_CONF[domain]["input_adapter"](stride_level=DOMAIN_CONF[domain]["stride_level"], patch_size_full=args.patch_size,
                                                                   image_size=(args.img_height, args.img_width)) for domain in ["rgb", "density"]}
    output_adapters = {domain: DOMAIN_CONF[domain]["output_adapter"](stride_level=DOMAIN_CONF[domain]["stride_level"], patch_size_full=args.patch_size, dim_tokens=args.dim_tokens,
                       depth=args.depth, use_task_queries=True, task=domain, context_tasks=DOMAINS, image_size=(args.img_height, args.img_width)) for domain in ["density"]}
    # model
    fuse_module = TransFuse(stride_level=args.stride_level, patch_size_full=args.patch_size, image_size=(args.img_height, args.img_width), num_channels=args.num_channels,
                            dim_tokens_enc=args.dim_tokens_enc, dim_tokens=args.dim_tokens, num_heads=args.num_heads)
    emac = e_mac(input_adapters=input_adapters, output_adapters=output_adapters, fuse_module=fuse_module, num_global_tokens=args.num_global_tokens)
    idx = str(args.type_dataset)
    ckpt = torch.load(args.ckpt_dir, map_location="cpu")
    ckpt_model = ckpt
    print('Load ckpt from:', args.ckpt_dir)
    interpolate_pos_embed_multimae(emac, ckpt_model)
    msg = emac.load_state_dict(ckpt_model, strict=False)
    emac = emac.cuda().eval()
    # test loader
    test_data = buildDensityDataset(args.input_dir, split="test", image_size=None, max_images=None, dens_norm=False, random_flip=False, dataset_name=args.type_dataset,
                                    clip_size=args.clip_size, stride=args.stride)
    gt = []
    pred = []
    for i in range(len(test_data)):
        data = test_data[i]
        imgs_rgb = data["rgb"] # [3, 2, 1024, 2048]
        den = data["density"] # [1, 2, 1024, 2048]
        name = data["name"]
        den = den.numpy().astype(np.float32) # [1, 2, 1024, 2048]
        gt_count = den[:, -1].sum()
        imgs_rgb, hc, wc = make_img_crop(imgs_rgb, args.img_height, args.img_width) # [18, 3, 2, 320, 320]
        input_dict = {}
        input_dict["rgb"] = imgs_rgb # [18, 3, 2, 320, 320]
        input_dict["density"] = torch.rand_like(input_dict["rgb"]).sum(dim=1, keepdim=True) # [18, 1, 2, 320, 320]
        input_dict = {k: v.cuda() for k, v in input_dict.items()}
        num_encoded_tokens = int(args.img_height // args.patch_size * args.img_width // args.patch_size)
        task_masks = {}
        task_masks["rgb"] = torch.zeros(imgs_rgb.shape[0], num_encoded_tokens).cuda() # [18, 400]
        task_masks["density"] = torch.ones(imgs_rgb.shape[0], num_encoded_tokens).cuda() # [18, 400]
        preds, masks, pred_fuse, img_warp, preds_prev, preds_prev_warp = emac.forward(input_dict, mask_inputs=True, num_encoded_tokens=num_encoded_tokens, task_masks=task_masks)
        den_pred = pred_fuse
        den_pred_denorm = den_pred * Normalization[args.type_dataset]["STD"] + Normalization[args.type_dataset]["MEAN"] # [18, 1, 320, 320]
        den_pred_denorm = rearrange(den_pred_denorm, "(hc wc) c h w -> c (hc h) (wc w)", hc=hc, wc=wc).detach().cpu() # [1, 960, 1920]
        pred_count = den_pred_denorm.sum()
        test_log = "{}: err: {:.4f}, gt_count: {:.4f}, pred_count: {:.4f}, name: {}".format(i, abs(gt_count - pred_count), gt_count, pred_count, name)
        print(test_log)
        with open(os.path.join(os.path.dirname(args.ckpt_dir), "{}_result.txt".format(idx)), "a") as f:
            f.write(test_log + "\n")
        gt.append(gt_count)
        pred.append(pred_count)
    test_log = "MAE: {:.2f}, RMSE: {:.2f}".format(mean_absolute_error(gt, pred), mean_squared_error(gt, pred, squared=False))
    print(test_log)
    with open(os.path.join(os.path.dirname(args.ckpt_dir), "{}_result.txt".format(idx)), "a") as f:
        f.write(test_log + "\n")
        f.write(args.ckpt_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', type=str, default='DroneBird')
    parser.add_argument('--input_dir', type=str, default='datasets/DroneBird')
    parser.add_argument('--img_height', type=int, default=320)
    parser.add_argument('--img_width', type=int, default=320)
    # model config
    parser.add_argument('--ckpt_dir', type=str, default='saved_dronebird/checkpoint-best.pth')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--dim_tokens', type=int, default=256)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--stride_level', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--dim_tokens_enc', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_global_tokens', type=int, default=1)
    # dataset config
    parser.add_argument('--clip_size', type=int, default=2)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()

    print('Testing dataset:', args.type_dataset)
    main(args)