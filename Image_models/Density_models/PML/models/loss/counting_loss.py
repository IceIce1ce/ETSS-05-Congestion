import einops
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from torch.nn.functional import interpolate
from torch import nn

def pt2dmap(pts, h, w, num_factor=1, sigma=0, scale=1):
    dmap = np.zeros((h, w))
    for pt in pts:
        dmap[min(int(pt[1].item() / scale), h - 1), min(int(pt[0].item() / scale), w - 1)] += num_factor
    if sigma > 0:
        dmap = gaussian_filter(dmap, sigma, mode="constant", cval=0)
    return dmap

def bbox2dmap(bboxes, h, w, num_factor=1, sigma=0, scale=1):
    dmap = np.zeros((h, w))
    for bbox in bboxes:
        left = round(max(bbox[0] / scale,0))
        right = round(min((bbox[0] + bbox[2]) / scale, w))
        top = round(max(bbox[1] / scale, 0))
        bottom = round(min((bbox[1] + bbox[3]) / scale, h))
        bottom = max(bottom, top + 1)
        right = max(right, left + 1)
        area= (right - left) * (bottom - top)
        dmap[top:bottom, left:right] += num_factor / area
    if sigma > 0:
        dmap = gaussian_filter(dmap, sigma)
    return dmap

class Cascade_MSE_Loss(torch.nn.Module):
    def __init__(self, scales=[1, 2, 4, 8, 16]):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        B, _, H, W = predict.shape
        target_maps = []
        predict_maps = []
        for i in range(self.num_scales):
            d = self.scales[i]
            predict_maps.append(einops.reduce(predict, "B C (h1 h2) (w1 w2)->B C h1 w1", "sum", h1=d, w1=d))
            target_maps.append(einops.reduce(target, "B C (h1 h2) (w1 w2)->B C h1 w1", "sum", h1=d, w1=d))
        mse_dict = {}
        index_dict = {}
        loss_dict = {}
        for i in range(self.num_scales):
            keyname = "scale_{}".format(self.scales[i])
            mse_dict[keyname] = self.mse(target_maps[i], predict_maps[i]) / B
            index_dict[keyname] = i
        for i in range(self.num_scales):
            if i == 0:
                cur_mse = mse_dict["scale_{}".format(self.scales[i])]
                diff = 1
                loss_dict["scale_{}".format(self.scales[i])] = torch.log(cur_mse + 1) * diff
            else:
                cur_mse = mse_dict["scale_{}".format(self.scales[i])]
                prev_mse = mse_dict["scale_{}".format(self.scales[i - 1])]
                ratio = (self.scales[i - 1] / self.scales[i]) ** 2
                diff = 1
                mse_change = F.relu(cur_mse - ratio * prev_mse)
                loss_dict["scale_{}".format(self.scales[i])] = torch.log(mse_change+1) * diff
        loss = sum(loss_dict.values())
        return loss

class Cascade_Weighted_MAE_Loss(torch.nn.Module):
    def __init__(self,scale_list=[1,2,4]) -> None:
        super().__init__()
        self.scale_list = scale_list
        self.eps = 1e-8

    def forward(self,predict,target):
        B, _, H, W = predict.shape
        loss = 0
        normalized_pre_mae = torch.ones_like(predict).requires_grad_(False)
        for idx, scale in enumerate(self.scale_list):
            target_map = einops.reduce(predict,"B C (h1 h2) (w1 w2)->B C h1 w1","mean", h1=scale, w1=scale)
            predict_map = einops.reduce(target,"B C (h1 h2) (w1 w2)->B C h1 w1","mean", h1=scale, w1=scale)
            normalized_pre_mae = interpolate(normalized_pre_mae, size=(scale, scale), mode="bilinear", align_corners=True).requires_grad_(False)
            if idx > 0:
                loss += (torch.abs(target_map - predict_map) * normalized_pre_mae).mean()
            
            with torch.no_grad():
                pre_mae = torch.exp(torch.abs(target_map - predict_map).requires_grad_(False))
                normalized_pre_mae = pre_mae / (self.eps + torch.sum(pre_mae, dim=(1, 2, 3), keepdim=True)) * (scale * scale)
        return loss

class Cascade_MAE_Loss(torch.nn.Module):
    def __init__(self, scale_list=[1,2,4]) -> None:
        super().__init__()
        self.scale_list = scale_list
        self.loss = torch.nn.L1Loss(reduction="mean")

    def forward(self, predict, target):
        B, _, H, W = predict.shape
        target_maps = []
        predict_maps = []
        for scale in self.scale_list:
            target_maps.append(einops.reduce(predict, "B C (h1 h2) (w1 w2)->B C h1 w1", "mean", h1=scale, w1=scale))
            predict_maps.append(einops.reduce(target, "B C (h1 h2) (w1 w2)->B C h1 w1", "mean", h1=scale, w1=scale))
        loss = 0
        for idx,sclae in enumerate(self.scale_list):
            if idx == 0:
                loss+= self.loss(target_maps[idx], predict_maps[idx])
            else:
                tar_map = target_maps[idx] - F.interpolate(target_maps[idx - 1], scale_factor=2)
                prd_map = predict_maps[idx] - F.interpolate(predict_maps[idx - 1], scale_factor=2)
                loss += self.loss(tar_map, prd_map)
        return loss

class DenseMap_Loss(torch.nn.Module):
    def __init__(self, cfg) -> None:
        self.kernel_size = cfg.kernel_size
        self.number_factor = cfg.num_factor
        super(DenseMap_Loss, self).__init__()
        self.loss = {"mse": torch.nn.MSELoss(reduction='none'), "mae": torch.nn.L1Loss(reduction='none'), "scale_mae": Cascade_MAE_Loss(cfg.scale_list),
                     "cascade_mae": Cascade_Weighted_MAE_Loss(cfg.scale_list), "cascade_mse": Cascade_MSE_Loss(cfg.scale_list)}
        self.weight_dict = {"mae": cfg.mae_weight, "mse": cfg.mse_weight, "scale_mae": cfg.scale_mae_weight, "cascade_mae": cfg.casecade_mae_weight,
                            "cascade_mse": cfg.casecade_mse_weight, "ot": cfg.ot_weight}
        self.resize = cfg.resize_to_original

    def forward(self, predict, targets, scale): # predict: [8, 1, 128, 128], scale: 8
        if self.resize:
            predict = F.interpolate(predict, scale_factor=scale)
        dMaps = np.zeros((predict.shape), dtype=np.float32) # [8, 1, 128, 128]
        with torch.no_grad():
            for idx in range(predict.shape[0]):
                if "bboxes" in targets: 
                    dMaps[idx, 0, ...] = bbox2dmap(targets["bboxes"][idx][:targets["num"][idx]], dMaps.shape[-2], dMaps.shape[-1], self.number_factor, self.kernel_size, scale)
                else:
                    dMaps[idx, ...] = pt2dmap(targets["points"][idx][:targets["num"][idx]], dMaps.shape[-2], dMaps.shape[-1], self.number_factor, self.kernel_size,scale)
        dMaps_cuda = torch.from_numpy(dMaps).cuda(predict.device)
        loss_dict = {k: torch.mean(self.loss[k](predict, dMaps_cuda)) for k in self.loss if k != "ot" and k != "cascade_mse" and  self.weight_dict[k] > 0}
        if self.weight_dict["cascade_mse"] > 0:
            loss_dict["cascade_mse"] = self.loss["cascade_mse"](predict, dMaps_cuda)
        loss_dict["all"] = sum([loss_dict[k]*self.weight_dict[k] for k in self.loss if self.weight_dict[k] > 0])
        return loss_dict, dMaps