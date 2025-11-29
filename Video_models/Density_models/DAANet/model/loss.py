import torch.nn.functional as F
from model.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
import torch

class ComputeKPILoss(object):
    def __init__(self, trainer, cfg) -> None:
        self.cfg = cfg
        self.trainer = trainer
        self.den_factor = cfg.den_factor
        self.scale_weight = cfg.scale_weight
        self.dynamic_weight = []

    def __call__(self, den, den_scales, gt_den_scales, masks, gt_mask, pre_outflow_map, pre_inflow_map, gt_io_map, pre_inf_cnt, pre_out_cnt, gt_in_cnt, gt_out_cnt, confidence):
        img_pair_num = den_scales[0].shape[0] // 2
        assert den.shape == gt_den_scales[0].shape
        self.cnt_loss = F.mse_loss(den * self.den_factor, gt_den_scales[0] * self.den_factor)
        self.cnt_loss_scales = torch.zeros(len(den_scales)).cuda()
        for scale in range(len(den_scales)):
            assert den_scales[scale].shape == gt_den_scales[scale].shape
            weight = F.adaptive_avg_pool2d(confidence[:,scale,:,:].unsqueeze(1), den_scales[scale].shape[2:])
            self.cnt_loss_scales[scale] += F.mse_loss(den_scales[scale] * self.den_factor, weight * gt_den_scales[scale] * self.den_factor) * self.scale_weight[scale]
        scale_loss = self.cnt_loss_scales.sum()
        self.mask_loss = F.binary_cross_entropy(masks[:img_pair_num], gt_mask[:, 0:1, :, :],reduction = "mean") + F.binary_cross_entropy(masks[img_pair_num:], gt_mask[:, 1:2, :, :],reduction = "mean")
        assert (pre_outflow_map.shape == gt_io_map[:, 0:1, :, :].shape) and (pre_inflow_map.shape == gt_io_map[:, 1:2, :, :].shape)
        self.out_loss = F.mse_loss(pre_outflow_map,  gt_io_map[:, 0:1, :, :],reduction = 'sum') / self.cfg.train_batch_size
        self.in_loss = F.mse_loss(pre_inflow_map, gt_io_map[:, 1:2, :, :], reduction='sum') / self.cfg.train_batch_size
        if self.trainer.i_tb == 1:
            self.init_scale_loss = scale_loss.item()
        elif self.trainer.resume:
            self.init_scale_loss = 0.3
        self.dynamic_weight.append((self.init_scale_loss - scale_loss.item()) / (self.init_scale_loss+1e-16))
        if self.trainer.i_tb > 1000 and len(self.dynamic_weight) > 1000:
            self.dynamic_weight.pop(0)
            assert len(self.dynamic_weight) == 1000
        avg_dynamic_weight = sum(self.dynamic_weight) / len(self.dynamic_weight)
        loss = scale_loss  + avg_dynamic_weight * (self.cnt_loss * self.cfg.cnt_weight) + (self.out_loss + self.in_loss) * self.cfg.io_weight + self.mask_loss * self.cfg.mask_weight
        return loss

    def compute_con_loss(self, pair_idx, feature1, feature2, match_gt, pois, count_in_pair, feature_scale):
        mdesc0, mdesc1 = self.get_head_feature(pair_idx, feature1, feature2, pois, count_in_pair, feature_scale)
        con_inter_loss = self.contrastive_loss(mdesc0, mdesc1, match_gt['a2b'][:,0], match_gt['a2b'][:,1])
        return con_inter_loss.sum()

    def get_head_feature(self, pair_idx, feature1, feature2, pois, count_in_pair, feature_scale):
        feature = torch.cat([feature1[pair_idx:pair_idx + 1], feature2[pair_idx:pair_idx + 1]], dim=0)
        poi_features = prroi_pool2d(feature, pois, 1, 1, feature_scale)
        poi_features = poi_features.squeeze(2).squeeze(2)[None].transpose(1, 2)
        mdesc0, mdesc1 = torch.split(poi_features, count_in_pair, dim=2)
        return mdesc0, mdesc1
    
    def contrastive_loss(self, mdesc0, mdesc1, idx0, idx1):
        sim_matrix = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        m0 = torch.norm(mdesc0, dim=1)
        m1 = torch.norm(mdesc1, dim=1)
        norm = torch.einsum('bn,bm->bnm', m0, m1) + 1e-7
        exp_term = torch.exp(sim_matrix / (256 ** .5) / norm)[0]
        try:
            topk = torch.topk(exp_term[idx0], 50, dim=1).values
        except:
            topk = exp_term[idx0]
        denominator = torch.sum(topk, dim=1)
        numerator = exp_term[idx0, idx1]
        loss = torch.sum(-torch.log(numerator / denominator + 1e-7))
        return loss