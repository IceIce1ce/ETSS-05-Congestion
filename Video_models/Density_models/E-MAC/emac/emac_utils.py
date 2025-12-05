import math
import warnings
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from functools import partial
from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as TF

def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(img.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def warp(x, flo): # [6, 3, 512, 512], [6, 2, 512, 512]
    B, C, H, W = x.size()
    device = x.device
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(device)
    vgrid = grid + flo
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1).to(device)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size(), device=device)
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask # [6, 3, 512, 512]

def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert (embed_dim % 4 == 0), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w, d=embed_dim)
    return pos_emb

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class TransFuse(nn.Module):
    def __init__(self, num_channels: int, stride_level: int, patch_size_full: Union[int, Tuple[int, int]], dim_tokens_enc: Optional[int] = None, dim_tokens: int = 256,
                 image_size: Union[int, Tuple[int]] = 224, mlp_ratio: int = 4.0, num_heads: int = 8, qkv_bias: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens_enc = dim_tokens_enc
        self.dim_tokens = dim_tokens
        self.image_size = pair(image_size)
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
        self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)
        self.decoder = CrossAttention(dim=self.dim_tokens, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.context_norm = norm_layer(self.dim_tokens)
        self.query_norm = norm_layer(self.dim_tokens)
        self.out_norm = norm_layer(self.dim_tokens)
        mlp_hidden_dim = int(self.dim_tokens * mlp_ratio)
        self.mlp = Mlp(in_features=self.dim_tokens, hidden_features=mlp_hidden_dim)
        self.dim_patch = self.num_channels * self.P_H * self.P_W
        self.out_proj = nn.Linear(self.dim_tokens, self.dim_patch)
        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        self.dim_tokens_enc = dim_tokens_enc
        self.proj_context = nn.Linear(self.dim_tokens_enc, self.dim_tokens)

    def forward(self, prev_tokens: torch.Tensor, cur_tokens: torch.Tensor): # [6, 1024, 768], [6, 1024, 768]
        assert (self.dim_tokens_enc is not None), "Need to call init(dim_tokens_enc) function first"
        H, W = self.image_size
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        prev_tokens = self.proj_context(prev_tokens)
        cur_tokens = self.proj_context(cur_tokens)
        pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bilinear", align_corners=False)
        pos_emb = rearrange(pos_emb, "b d nh nw -> b (nh nw) d")
        prev_tokens = prev_tokens + pos_emb
        queries_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bilinear", align_corners=False)
        queries_pos_emb = rearrange(queries_pos_emb, "b d nh nw -> b (nh nw) d")
        cur_tokens = cur_tokens + queries_pos_emb
        cur_tokens = self.decoder(self.query_norm(cur_tokens), self.context_norm(prev_tokens))
        cur_tokens = cur_tokens + self.mlp(self.out_norm(cur_tokens))
        cur_tokens = self.out_proj(cur_tokens)
        residual = rearrange(cur_tokens, "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)", nh=N_H, nw=N_W, ph=self.P_H, pw=self.P_W, c=self.num_channels) # [6, 1, 512, 512]
        return residual

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x