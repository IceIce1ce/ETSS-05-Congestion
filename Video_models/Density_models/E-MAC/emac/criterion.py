import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x): # [6, 1, 512, 512]
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

@torch.cuda.amp.autocast(enabled=False)
class MaskedMSELoss(nn.Module):
    def __init__(self, patch_size: int = 16, stride: int = 1):
        super().__init__()
        self.scale_factor = patch_size // stride

    def forward(self, input, target, mask=None): # [6, 1, 512, 512], [6, 1, 512, 512]
        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor
        loss = F.mse_loss(input, target, reduction="none")
        with torch.cuda.amp.autocast(enabled=False):
            if mask is not None:
                if mask.sum() == 0:
                    return torch.tensor(0).to(loss.device)
                mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
                mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1)
                loss_float = loss.float()
                loss_float = loss_float.mean(dim=1)
                loss_float = loss_float * mask
                loss_float = loss_float.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
                loss_float = loss_float.nanmean()
                loss = loss_float
            else:
                loss = loss.mean()
        return loss