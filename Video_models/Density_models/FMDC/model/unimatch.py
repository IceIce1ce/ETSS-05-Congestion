# https://github.com/haofeixu/gmflow/blob/main/gmflow/gmflow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow, global_correlation_softmax_stereo, local_correlation_softmax_stereo, correlation_softmax_depth
from .attention import SelfAttnPropagation
from .geometry import flow_warp, compute_flow_with_depth_pose
from .reg_refine import BasicUpdateBlock
from .utils import normalize_img, feature_add_position, upsample_flow_with_mask

class UniMatch(nn.Module):
    def __init__(self, num_scales=1, feature_channels=128, upsample_factor=8, num_head=1, ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=False, task='flow'):
        super(UniMatch, self).__init__()
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers, d_model=feature_channels, nhead=num_head, ffn_dim_expansion=ffn_dim_expansion)
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)
        if not self.reg_refine or task == 'depth':
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1), nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        if reg_refine:
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2, downsample_factor=upsample_factor, flow_dim=2 if task == 'flow' else 1, bilinear_up=task == 'depth')

    def extract_feature(self, img0, img1): # [2, 3, 192, 256], [2, 3, 192, 256]
        concat = torch.cat((img0, img1), dim=0)
        features = self.backbone(concat)
        features = features[::-1]
        feature0, feature1 = [], []
        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0) # [2, 128, 24, 32], [2, 128, 24, 32]
            feature0.append(chunks[0])
            feature1.append(chunks[1])
        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8, is_depth=False): # [2, 2, 24, 32], None
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor, mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor, is_depth=is_depth)
        return up_flow # [2, 2, 192, 256]

    def forward(self, img0, img1, attn_type=None, attn_splits_list=None, corr_radius_list=None, prop_radius_list=None, num_reg_refine=1, pred_bidir_flow=False, task='flow',
                intrinsics=None, pose=None, min_depth=1. / 0.5, max_depth=1. / 10, num_depth_candidates=64, depth_from_argmax=False, pred_bidir_depth=False, **kwargs):
        if pred_bidir_flow:
            assert task == 'flow'
        if task == 'depth':
            assert self.num_scales == 1
        results_dict = {}
        flow_preds = []
        if task == 'flow':
            img0, img1 = normalize_img(img0, img1)
        feature0_list, feature1_list = self.extract_feature(img0, img1)
        flow = None
        if task != 'depth':
            assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        else:
            assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1
        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
            if pred_bidir_flow and scale_idx > 0:
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)
            feature0_ori, feature1_ori = feature0, feature1
            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))
            if task == 'depth':
                intrinsics_curr = intrinsics.clone()
                intrinsics_curr[:, :2] = intrinsics_curr[:, :2] / upsample_factor
            if scale_idx > 0:
                assert task != 'depth'
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
            if flow is not None:
                assert task != 'depth'
                flow = flow.detach()
                if task == 'stereo':
                    zeros = torch.zeros_like(flow)
                    displace = torch.cat((-flow, zeros), dim=1)
                    feature1 = flow_warp(feature1, displace)
                elif task == 'flow':
                    feature1 = flow_warp(feature1, flow)
                else:
                    raise NotImplementedError
            attn_splits = attn_splits_list[scale_idx]
            if task != 'depth':
                corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            feature0, feature1 = self.transformer(feature0, feature1, attn_type=attn_type, attn_num_splits=attn_splits)
            if task == 'depth':
                b, _, h, w = feature0.size()
                depth_candidates = torch.linspace(min_depth, max_depth, num_depth_candidates).type_as(feature0)
                depth_candidates = depth_candidates.view(1, num_depth_candidates, 1, 1).repeat(b, 1, h, w)
                flow_pred = correlation_softmax_depth(feature0, feature1, intrinsics_curr, pose, depth_candidates=depth_candidates, depth_from_argmax=depth_from_argmax, pred_bidir_depth=pred_bidir_depth)[0]
            else:
                if corr_radius == -1:
                    if task == 'flow':
                        flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
                    elif task == 'stereo':
                        flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
                    else:
                        raise NotImplementedError
                else:
                    if task == 'flow':
                        flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]
                    elif task == 'stereo':
                        flow_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]
                    else:
                        raise NotImplementedError
            flow = flow + flow_pred if flow is not None else flow_pred
            if task == 'stereo':
                flow = flow.clamp(min=0)
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor, is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)
            if (pred_bidir_flow or pred_bidir_depth) and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)
            flow = self.feature_flow_attn(feature0, flow.detach(), local_window_attn=prop_radius > 0, local_window_radius=prop_radius)
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor, is_depth=task == 'depth')
                flow_preds.append(flow_up)
            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    if task == 'stereo':
                        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)
                        flow_up_pad = self.upsample_flow(flow_pad, feature0)
                        flow_up = -flow_up_pad[:, :1]
                    elif task == 'depth':
                        depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)
                        depth_up_pad = self.upsample_flow(depth_pad, feature0, is_depth=True).clamp(min=min_depth, max=max_depth)
                        flow_up = depth_up_pad[:, :1]
                    else:
                        flow_up = self.upsample_flow(flow, feature0)
                    flow_preds.append(flow_up)
                else:
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor, is_depth=task == 'depth')
                        flow_preds.append(flow_up)
                    assert num_reg_refine > 0
                    for refine_iter_idx in range(num_reg_refine):
                        flow = flow.detach()
                        if task == 'stereo':
                            zeros = torch.zeros_like(flow)
                            displace = torch.cat((-flow, zeros), dim=1)
                            correlation = local_correlation_with_flow(feature0_ori, feature1_ori, flow=displace, local_radius=4)
                        elif task == 'depth':
                            if pred_bidir_depth and refine_iter_idx == 0:
                                intrinsics_curr = intrinsics_curr.repeat(2, 1, 1)
                                pose = torch.cat((pose, torch.inverse(pose)), dim=0)
                                feature0_ori, feature1_ori = torch.cat((feature0_ori, feature1_ori), dim=0), torch.cat((feature1_ori, feature0_ori), dim=0)
                            flow_from_depth = compute_flow_with_depth_pose(1. / flow.squeeze(1), intrinsics_curr, extrinsics_rel=pose)
                            correlation = local_correlation_with_flow(feature0_ori, feature1_ori, flow=flow_from_depth, local_radius=4)
                        else:
                            correlation = local_correlation_with_flow(feature0_ori, feature1_ori, flow=flow, local_radius=4)
                        proj = self.refine_proj(feature0)
                        net, inp = torch.chunk(proj, chunks=2, dim=1)
                        net = torch.tanh(net)
                        inp = torch.relu(inp)
                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone())
                        if task == 'depth':
                            flow = (flow - residual_flow).clamp(min=min_depth, max=max_depth)
                        else:
                            flow = flow + residual_flow
                        if task == 'stereo':
                            flow = flow.clamp(min=0)
                        if self.training or refine_iter_idx == num_reg_refine - 1:
                            if task == 'depth':
                                if refine_iter_idx < num_reg_refine - 1:
                                    flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor, is_depth=True)
                                else:
                                    depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)
                                    depth_up_pad = self.upsample_flow(depth_pad, feature0, is_depth=True).clamp(min=min_depth, max=max_depth)
                                    flow_up = depth_up_pad[:, :1]
                            else:
                                flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor, is_depth=task == 'depth')
                            flow_preds.append(flow_up)
        if task == 'stereo':
            for i in range(len(flow_preds)):
                flow_preds[i] = flow_preds[i].squeeze(1)
        if task == 'depth':
            for i in range(len(flow_preds)):
                flow_preds[i] = 1. / flow_preds[i].squeeze(1)
        results_dict.update({'flow_preds': flow_preds}) # [2, 2, 192, 256]
        return results_dict