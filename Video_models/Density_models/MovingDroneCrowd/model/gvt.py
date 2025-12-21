import torch.nn as nn
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block as TimmBlock
import math

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, f"img_size {(H, W)} should be divided by patch_size {self.patch_size}."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, H, W
    
class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([TimmBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)]) # new timm version: set qk_scale=None
        self.norm = norm_layer(embed_dim)
        self.init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.apply(self._init_weights)
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H, W): # [2, 3072, 256]
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2) # [2, 3072, 256]
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class CViTV2(VisionTransformer):
    def __init__(self, patch_size=16, in_chans=3, embed_dims=256, depths=12, num_heads=12, mlp_ratios=4., norm_layer=nn.LayerNorm):
        super(CViTV2, self).__init__(patch_size, in_chans, embed_dims, depths, num_heads, mlp_ratios, norm_layer)
        self.pos_block = PosCNN(embed_dims, embed_dims)
        self.norm = norm_layer(embed_dims)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def no_weight_decay(self):
        return set(['cls_token'] + ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def forward_features(self, x): # [2, 3, 768, 1024]
        outputs = list()
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i == 0:
                x = self.pos_block(x, H, W)
            outputs.append(self.norm(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        return outputs # [2, 256, 48, 64]

    def forward(self, x):
        x = self.forward_features(x)
        return x

class pcvit_base(CViTV2):
    def __init__(self, **kwargs):
        super(pcvit_base, self).__init__(patch_size=16, embed_dims=256, num_heads=8, depths=6, mlp_ratios=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))