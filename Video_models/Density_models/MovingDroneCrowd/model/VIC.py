from functools import partial
from .gvt import pcvit_base
import torch.nn as nn
import torch
from misc.layer import Gaussianlayer
from model.ViT.models_crossvit import CrossAttentionBlock, FeatureFusionModule
from model.VGG.VGG16_FPN import VGG16_FPN_Encoder
from model.decoder import ShareDecoder, InOutDecoder, GlobalDecoder
from model.ResNet.ResNet50_FPN import ResNet_50_FPN_Encoder

class Video_Counter(nn.Module):
    def __init__(self, cfg, cfg_data):
        super(Video_Counter, self).__init__()
        self.cfg = cfg
        self.cfg_data = cfg_data
        if cfg.encoder == 'VGG16_FPN':
            self.Extractor = VGG16_FPN_Encoder()
        elif cfg.encoder == 'ResNet_50_FPN':
            self.Extractor = ResNet_50_FPN_Encoder()
        elif cfg.encoder == 'PCPVT':
            self.Extractor = pcvit_base()
        else:
            print('This encoder does not exist')
            raise NotImplementedError
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.share_cross_attention = nn.ModuleList([nn.ModuleList([CrossAttentionBlock(cfg.cross_attn_embed_dim, cfg.cross_attn_num_heads, cfg.mlp_ratio, qkv_bias=True, qk_scale=None,
                                                                                       norm_layer=norm_layer) for _ in range(cfg.cross_attn_depth)]) for _ in range(3)])
        self.share_cross_attention_norm = norm_layer(cfg.cross_attn_embed_dim)
        self.feature_fuse = FeatureFusionModule(self.cfg.FEATURE_DIM)
        self.global_decoder = GlobalDecoder()
        self.share_decoder = ShareDecoder()
        self.in_out_decoder = InOutDecoder()
        self.criterion = torch.nn.MSELoss()
        self.Gaussian = Gaussianlayer()
        
    def forward(self, img, target): # [2, 3, 768, 1024]
        features = self.Extractor(img) # [2, 256, 48, 64] * len(4)
        B, C, H, W = features[-1].shape
        self.feature_scale = H / img.shape[2] 
        pre_global_den = self.global_decoder(features[-1]) # [2, 1, 768, 1024]
        all_loss = {}
        gt_in_out_dot_map = torch.zeros_like(pre_global_den)
        gt_share_dot_map = torch.zeros_like(pre_global_den)
        img_pair_num = img.size(0) // 2
        assert img.size(0) % 2 == 0
        share_features = None
        for l_num in range(len(self.share_cross_attention)):
            share_results = []
            if share_features is not None:
                feature_fused = self.feature_fuse(share_features, features[l_num]) # [2, 256, 48, 64]
            for pair_idx in range(img_pair_num):
                if share_features is not None:
                    q1 = feature_fused[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                else:
                    q1 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                kv = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                for i in range(len(self.share_cross_attention[l_num])):
                    q1 = self.share_cross_attention[l_num][i](q1, kv) # [1, 3072, 256]
                q1 = self.share_cross_attention_norm(q1) # [1, 3072, 256]
                if share_features is not None:
                    q2 = feature_fused[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                else:
                    q2 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                kv = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                for i in range(len(self.share_cross_attention[l_num])):
                    q2 = self.share_cross_attention[l_num][i](q2, kv)
                q2 = self.share_cross_attention_norm(q2)
                share_results.append(q1)
                share_results.append(q2)
            share_features = torch.cat(share_results, dim=0) # [2, 3072, 256]
            share_features = share_features.permute(0, 2, 1).reshape(B, C, H, W).contiguous() # [2, 256, 48, 64]
        for pair_idx in range(img_pair_num):
            points0 = target[pair_idx * 2]['points']
            points1 = target[pair_idx * 2 + 1]['points']
            share_mask0 = target[pair_idx * 2]['share_mask0']
            outflow_mask = target[pair_idx * 2]['outflow_mask']
            share_mask1 = target[pair_idx * 2 + 1]['share_mask1']
            inflow_mask = target[pair_idx * 2 + 1]['inflow_mask']
            share_coords0 = points0[share_mask0].long()
            share_coords1 = points1[share_mask1].long()
            gt_share_dot_map[pair_idx * 2, 0, share_coords0[:, 1], share_coords0[:, 0]] = 1
            gt_share_dot_map[pair_idx * 2 + 1, 0, share_coords1[:, 1], share_coords1[:, 0]] = 1
            outflow_coords = points0[outflow_mask].long()
            inflow_coords = points1[inflow_mask].long()
            gt_in_out_dot_map[pair_idx * 2, 0, outflow_coords[:, 1], outflow_coords[:, 0]] = 1
            gt_in_out_dot_map[pair_idx * 2 + 1, 0, inflow_coords[:, 1], inflow_coords[:, 0]] = 1
        pre_share_den = self.share_decoder(share_features) # [2, 1, 768, 1024]
        mid_pre_in_out_den = pre_global_den - pre_share_den
        pre_in_out_den = self.in_out_decoder(mid_pre_in_out_den) # [2, 1, 768, 1024]
        # density map loss
        gt_global_dot_map = torch.zeros_like(pre_global_den)
        for i, data in enumerate(target):
            points = data['points'].long()
            gt_global_dot_map[i, 0, points[:, 1], points[:, 0]] = 1
        gt_global_den = self.Gaussian(gt_global_dot_map) # [2, 1, 768, 1024]
        assert pre_global_den.size() == gt_global_den.size()
        global_mse_loss = self.criterion(pre_global_den, gt_global_den * self.cfg_data.DEN_FACTOR)
        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['global'] = global_mse_loss
        gt_share_den = self.Gaussian(gt_share_dot_map) # [2, 1, 768, 1024]
        assert pre_share_den.size() == gt_share_den.size()
        share_mse_loss = self.criterion(pre_share_den, gt_share_den * self.cfg_data.DEN_FACTOR)
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['share'] = share_mse_loss * 10
        gt_in_out_den = self.Gaussian(gt_in_out_dot_map) # [2, 1, 768, 1024]
        assert pre_in_out_den.size() == gt_in_out_den.size()
        in_out_mse_loss = self.criterion(pre_in_out_den, gt_in_out_den * self.cfg_data.DEN_FACTOR)
        pre_in_out_den = pre_in_out_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['in_out'] = in_out_mse_loss
        return pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss
    
    def test_forward(self, img):
        features = self.Extractor(img)
        B, C, H, W = features[-1].shape
        pre_global_den = self.global_decoder(features[-1])
        img_pair_num = img.size(0) // 2
        assert img.size(0) % 2 == 0
        share_features = None
        for l_num in range(len(self.share_cross_attention)):
            share_results = []
            if share_features is not None:
                feature_fused = self.feature_fuse(share_features, features[l_num])
            for pair_idx in range(img_pair_num):
                if share_features is not None:
                    q1 = feature_fused[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                else:
                    q1 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                kv = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                for i in range(len(self.share_cross_attention[l_num])):
                    q1 = self.share_cross_attention[l_num][i](q1, kv)
                q1 = self.share_cross_attention_norm(q1)
                if share_features is not None:
                    q2 = feature_fused[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                else:
                    q2 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                kv = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                for i in range(len(self.share_cross_attention[l_num])):
                    q2 = self.share_cross_attention[l_num][i](q2, kv)
                q2 = self.share_cross_attention_norm(q2)
                share_results.append(q1)
                share_results.append(q2)
            share_features = torch.cat(share_results, dim=0)
            share_features = share_features.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        pre_share_den = self.share_decoder(share_features)
        mid_pre_in_out_den = pre_global_den - pre_share_den
        pre_in_out_den = self.in_out_decoder(mid_pre_in_out_den)
        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        pre_in_out_den = pre_in_out_den.detach() / self.cfg_data.DEN_FACTOR
        return pre_global_den, pre_share_den, pre_in_out_den
