import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcn import DeformableConv2d
from .VGG.conv import ResBlock

class optical_deformable_alignment_module(nn.Module):
    def __init__(self):
        super(optical_deformable_alignment_module, self).__init__()
        self.offset_groups = 4
        self.deformable_kernel_size = 3
        self.padding = (self.deformable_kernel_size - 1) // 2
        self.deformable_conv1 = DeformableConv2d(256, 256, self.offset_groups, kernel_size=self.deformable_kernel_size, padding = self.padding)
        self.deformable_conv2 = DeformableConv2d(256, 256, self.offset_groups, kernel_size=self.deformable_kernel_size, padding = self.padding)
        self.weight_conv = ResBlock(512, 512)
        self.reduce_channel2 = ResBlock(512, 256)
        self.reduce_channel = ResBlock(256, 128)
        self.offset_loss = 0.
        
    def forward(self, reference, source): # [2, 256, 192, 256], [2, 256, 192, 256]
        ref_refined_feature, offset2sou = self.deformable_conv1(reference, source)
        sour_refined_feature, offset2ref = self.deformable_conv1(source, reference)
        offset2s = F.interpolate(offset2sou,scale_factor=4, mode='bilinear', align_corners=True) * 4 # [2, 2, 768, 1024]
        offset2r = F.interpolate(offset2ref,scale_factor=4, mode='bilinear', align_corners=True) * 4 # [2, 2, 768, 1024]
        self.offset_loss = self.deformable_conv1.offset_loss
        refcorsou = torch.concat([sour_refined_feature, reference], axis = 1)
        soucorref = torch.concat([ref_refined_feature, source], axis = 1)
        compare = torch.concat([refcorsou, soucorref], axis = 0)
        comp = self.weight_conv(compare)
        com = self.reduce_channel2(comp)
        compare_result = self.reduce_channel(com) # [4, 128, 192, 256]
        return compare_result, offset2r, offset2s

    def color(self, reference, flow, source): # [2, 256, 192, 256], [2, 2, 192, 256], [2, 256, 192, 256]
        pre_warp_ref = optical_flow_warping(reference, flow) # [2, 256, 192, 256]
        pre_refined_feature, _ = self.deformable_conv1(pre_warp_ref, source) # [2, 256, 192, 256]
        next_refined_feature, _ = self.deformable_conv2(source, pre_refined_feature) # [2, 256, 192, 256]
        weight_in = torch.concat([pre_refined_feature, next_refined_feature], axis=1) # [2, 512, 192, 256]
        weight_in = self.weight_conv(weight_in) # [2, 512, 192, 256]
        return weight_in

def optical_flow_warping(x, flo, pad_mode="border"): # [2, 256, 192, 256], [2, 2, 192, 256]
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().cuda()
    vgrid = grid + flo
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=pad_mode) # [2, 256, 192, 256]
    return output