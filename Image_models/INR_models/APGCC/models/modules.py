# https://github.com/tristandb/EfficientDet-PyTorch/blob/master/retinanet.py
# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1), nn.ReLU())
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=2, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Sequential(nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1), nn.ReLU())
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, _ = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class FPN(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, inner_planes=256, feat_layers=[1,2,3,4]):
        super(FPN, self).__init__()
        self.feat_layers = feat_layers
        if 4 in self.feat_layers:
            self.P5_1 = nn.Conv2d(C5_size, inner_planes, kernel_size=1, stride=1, padding=0)
            self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P5_2 = nn.Conv2d(inner_planes, inner_planes, kernel_size=3, stride=1, padding=1)
        if 3 in self.feat_layers:
            self.P4_1 = nn.Conv2d(C4_size, inner_planes, kernel_size=1, stride=1, padding=0)
            self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P4_2 = nn.Conv2d(inner_planes, inner_planes, kernel_size=3, stride=1, padding=1)
        if 2 in self.feat_layers:
            self.P3_1 = nn.Conv2d(C3_size, inner_planes, kernel_size=1, stride=1, padding=0) 
            self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P3_2 = nn.Conv2d(inner_planes, inner_planes, kernel_size=3, stride=1, padding=1)
        if 1 in self.feat_layers:
            self.P2_1 = nn.Conv2d(C2_size, inner_planes, kernel_size=1, stride=1, padding=0)
            self.P2_2 = nn.Conv2d(inner_planes, inner_planes, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs # [32, 128, 64, 64], [32, 256, 32, 32], [32, 512, 16, 16], [32, 512, 8, 8]
        output = []
        if 4 in self.feat_layers:
            P5_x = self.P5_1(C5)
            P5_upsampled_x = self.P5_upsampled(P5_x)
            P5_x = self.P5_2(P5_x)
            output.append(P5_x)
        if 3 in self.feat_layers:
            P4_x = self.P4_1(C4)
            P4_x = P5_upsampled_x + P4_x
            P4_upsampled_x = self.P4_upsampled(P4_x)
            P4_x = self.P4_2(P4_x)
            output.append(P4_x)
        if 2 in self.feat_layers:
            P3_x = self.P3_1(C3)
            P3_x = P3_x + P4_upsampled_x
            P3_upsampled_x = self.P3_upsampled(P3_x)
            P3_x = self.P3_2(P3_x)
            output.append(P3_x)
        if 1 in self.feat_layers:
            P2_x = self.P2_1(C2)
            P2_x = P2_x + P3_upsampled_x
            P2_x = self.P2_2(P2_x)
            output.append(P2_x)
        return output # [32, 256, 8, 8], [32, 256, 16, 16]

class ASPP(nn.Module):
    def __init__(self, in_planes, inner_planes=256, sync_bn=False, bn=False, dilations=(12, 24, 36)):
        super(ASPP, self).__init__()
        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm2d
        if bn == False:
            self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False), nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False), nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False), nn.ReLU(inplace=True))
            self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False), norm_layer(inner_planes), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False), norm_layer(inner_planes), nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False), norm_layer(inner_planes), nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False), norm_layer(inner_planes), nn.ReLU(inplace=True))
            self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False), norm_layer(inner_planes), nn.ReLU(inplace=True))
        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out

class ifi_simfpn(nn.Module):
    def __init__(self, ultra_pe=False, pos_dim=40, sync_bn=False, num_anchor_points=4, num_classes=2, local=False, unfold=False, 
                 stride=1, learn_pe=False, require_grad=False, head_layers=[512,256,256], feat_num=4, feat_dim=256):
        super(ifi_simfpn, self).__init__()
        self.pos_dim = pos_dim
        self.ultra_pe = ultra_pe
        self.local = local
        self.unfold = unfold
        self.stride = stride
        self.learn_pe = learn_pe
        self.feat_num = feat_num
        self.feat_dim = feat_dim
        self.num_anchor_points = num_anchor_points
        self.regression_dims = 2
        self.num_classes = num_classes
        self.head_layers = head_layers
        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm1d
        if learn_pe:
            for level in range(self.feat_num):
                self._update_property('pos' + str(level+1), PositionEmbeddingLearned(self.pos_dim//2))
        elif ultra_pe:
            for level in range(self.feat_num):
                self._update_property('pos' + str(level+1), SpatialEncoding(2, self.pos_dim, require_grad=require_grad))
            self.pos_dim += 2
        else:
            self.pos_dim = 2
        in_dim = self.feat_num*(self.feat_dim + self.pos_dim)
        if unfold:
            in_dim = self.feat_num*(self.feat_dim*9 + self.pos_dim)
        self.in_dim = in_dim
        confidence_head_list = []
        offset_head_list = []
        for ct, hidden_feature in enumerate(head_layers):
            if ct == 0:
                src_dim = in_dim
            else:
                src_dim = head_layers[ct-1]
            confidence_head_list.append([nn.Conv1d(src_dim, hidden_feature, 1), norm_layer(hidden_feature), nn.ReLU()])
            offset_head_list.append([nn.Conv1d(src_dim, hidden_feature, 1), norm_layer(hidden_feature), nn.ReLU()])
        confidence_head_list.append([nn.Conv1d(head_layers[-1], self.num_anchor_points*self.num_classes, 1), nn.ReLU()])
        offset_head_list.append([nn.Conv1d(head_layers[-1], self.num_anchor_points*2, 1)])
        confidence_head_list = [item for sublist in confidence_head_list for item in sublist]
        offset_head_list = [item for sublist in offset_head_list for item in sublist]
        self.confidence_head = nn.Sequential(*confidence_head_list)
        self.offset_head = nn.Sequential(*offset_head_list)

    def forward(self, x, size, level=0, after_cat=False):
        h, w = size
        if not after_cat:
            if not self.local:
                if self.unfold:
                    x = F.unfold(x, 3, padding=1).view(x.shape[0],x.shape[1]*9, x.shape[2], x.shape[3])
                rel_coord, q_feat = ifi_feat(x, [h, w]) # [32, 256, 2], [32, 256, 256]
                if self.ultra_pe:
                    rel_coord = eval('self.pos'+str(level))(rel_coord)
                elif self.learn_pe:
                    rel_coord = eval('self.pos'+str(level))(rel_coord, [1,1,h,w])
                x = torch.cat([rel_coord, q_feat], dim=-1) # [32, 256, 258]
            else:
                if self.unfold:
                    x = F.unfold(x, 3, padding=1).view(x.shape[0],x.shape[1]*9, x.shape[2], x.shape[3])
                rel_coord_list, q_feat_list, area_list = ifi_feat(x, [h, w],  local=True, stride=self.stride)
                total_area = torch.stack(area_list).sum(dim=0)
                context_list = []
                for rel_coord, q_feat, area in zip(rel_coord_list, q_feat_list, area_list):
                    if self.ultra_pe:
                        rel_coord = eval('self.pos'+str(level))(rel_coord)
                    elif self.learn_pe:
                        rel_coord = eval('self.pos'+str(level))(rel_coord, [1,1,h,w])
                    context_list.append(torch.cat([rel_coord, q_feat], dim=-1))
                ret = 0
                t = area_list[0]; area_list[0] = area_list[3]; area_list[3] = t
                t = area_list[1]; area_list[1] = area_list[2]; area_list[2] = t
                for conte, area in zip(context_list, area_list):
                    x = ret + conte *  ((area / total_area).unsqueeze(-1))          
            return x
        else:
            offset = self.offset_head(x).view(x.shape[0], -1, h, w)
            offset = offset.permute(0, 2, 3, 1)
            offset = offset.contiguous().view(x.shape[0], -1, 2)
            confidence = self.confidence_head(x).view(x.shape[0], -1, h, w)
            confidence = confidence.permute(0, 2, 3, 1)
            confidence = confidence.contiguous().view(x.shape[0], -1, self.num_classes)
            return offset, confidence

    def _update_property(self, property, value):
        setattr(self, property, value)

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(200, num_pos_feats)
        self.col_embed = nn.Embedding(200, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, shape):
        h, w = shape[2], shape[3]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1), y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).view(x.shape[0],h*w, -1)
        return pos

class SpatialEncoding(nn.Module):
    def __init__(self, in_dim, out_dim, sigma = 6, cat_input=True, require_grad=False,):
        super().__init__()
        assert out_dim % (2*in_dim) == 0, "dimension must be dividable"
        n = out_dim // 2 // in_dim
        m = 2**np.linspace(0, sigma, n)
        m = np.stack([m] + [np.zeros_like(m)]*(in_dim-1), axis=-1)
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        self.emb = torch.FloatTensor(m)
        if require_grad:
            self.emb = nn.Parameter(self.emb, requires_grad=True)    
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x):
        if not self.require_grad:
            self.emb = self.emb.to(x.device)
        y = torch.matmul(x, self.emb.T)
        if self.cat_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) # [16, 16, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) # [256, 2]
    return ret

def ifi_feat(res, size, stride=1, local=False): # [32, 256, 16, 16]
    bs, hh, ww = res.shape[0], res.shape[-2], res.shape[-1]
    h, w = size
    coords = (make_coord((h,w)).cuda().flip(-1) + 1) / 2
    coords = coords.unsqueeze(0).expand(bs, *coords.shape)
    coords = (coords*2-1).flip(-1)
    feat_coords = make_coord((hh,ww), flatten=False).cuda().permute(2, 0, 1) .unsqueeze(0).expand(res.shape[0], 2, *(hh,ww))
    if local:
        vx_list = [-1, 1]
        vy_list = [-1, 1]
        eps_shift = 1e-6
        rel_coord_list = []
        q_feat_list = []
        area_list = []
    else:
        vx_list, vy_list, eps_shift = [0], [0], 0
    rx = stride / h 
    ry = stride / w
    for vx in vx_list:
        for vy in vy_list:
            coords_ = coords.clone()
            coords_[:,:,0] += vx * rx + eps_shift
            coords_[:,:,1] += vy * ry + eps_shift
            coords_.clamp_(-1+1e-6, 1-1e-6)
            q_feat = F.grid_sample(res, coords_.flip(-1).unsqueeze(1),mode='nearest',align_corners=False)[:,:,0,:].permute(0,2,1)
            q_coord = F.grid_sample(feat_coords, coords_.flip(-1).unsqueeze(1),mode='nearest',align_corners=False)[:,:,0,:].permute(0,2,1)
            rel_coord = coords - q_coord
            rel_coord[:,:,0] *= hh
            rel_coord[:,:,1] *= ww
            if local:
                rel_coord_list.append(rel_coord)
                q_feat_list.append(q_feat)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                area_list.append(area+1e-9)
    if not local:
        return rel_coord, q_feat # [32, 256, 2], [32, 256, 256]
    else:
        return rel_coord_list, q_feat_list, area_list