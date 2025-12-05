from functools import partial
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .emac_utils import Block, CrossAttention, Mlp, build_2d_sincos_posemb, pair, trunc_normal_

class SpatialOutputAdapter(nn.Module):
    def __init__(self, num_channels: int, stride_level: int, patch_size_full: Union[int, Tuple[int, int]], dim_tokens_enc: Optional[int] = None, dim_tokens: int = 256,
                 depth: int = 0, learnable_pos_emb: int = False, image_size: Union[int, Tuple[int]] = 224, mlp_ratio: int = 4.0, num_heads: int = 8, qkv_bias: bool = True,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0, norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6), use_task_queries: bool = True,
                 task: Optional[str] = None, context_tasks: Optional[list] = None, use_xattn: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens_enc = dim_tokens_enc
        self.dim_tokens = dim_tokens
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.use_task_queries = use_task_queries
        self.task = task
        self.use_xattn = use_xattn
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)
        if context_tasks is not None:
            self.task_embeddings = nn.ParameterDict({task: nn.Parameter(torch.zeros(1, 1, self.dim_tokens)) for task in context_tasks})
            for embedding in self.task_embeddings.values():
                trunc_normal_(embedding, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if not self.learnable_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)
        if self.use_xattn:
            self.decoder = CrossAttention(dim=self.dim_tokens, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate)
            self.context_norm = norm_layer(self.dim_tokens)
            self.query_norm = norm_layer(self.dim_tokens)
            self.out_norm = norm_layer(self.dim_tokens)
            mlp_hidden_dim = int(self.dim_tokens * mlp_ratio)
            self.mlp = Mlp(in_features=self.dim_tokens, hidden_features=mlp_hidden_dim)
        if depth > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.decoder_transformer = nn.Sequential(*[Block(dim=self.dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                                             drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        else:
            self.decoder_transformer = nn.Identity()
        self.dim_patch = self.num_channels * self.P_H * self.P_W
        self.out_proj = nn.Linear(self.dim_tokens, self.dim_patch)
        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        self.dim_tokens_enc = dim_tokens_enc
        self.proj_context = nn.Linear(self.dim_tokens_enc, self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_emb", "mask_token", "task_embeddings"}

    def generate_context_embeddings(self, input_info, bs: int, size: Tuple[int, int], device: Optional[torch.device] = None):
        context_embeddings = []
        for task, info in input_info["tasks"].items():
            if self.task_embeddings is not None and task in self.task_embeddings:
                task_emb = repeat(self.task_embeddings[task], "() () d -> b n d", b=bs, n=info["num_tokens"])
            else:
                task_emb = torch.zeros((bs, info["num_tokens"], self.dim_tokens), device=device)
            if info["has_2d_posemb"]:
                pos_emb = F.interpolate(self.pos_emb, size=size, mode="bilinear", align_corners=False)
                pos_emb = rearrange(pos_emb, "b d nh nw -> b (nh nw) d")
                assert info["num_tokens"] == pos_emb.shape[1]
                task_emb = task_emb + pos_emb
            context_embeddings.append(task_emb)
        context_embeddings = torch.cat(context_embeddings, dim=1) # [6, 2048, 256]
        return context_embeddings

    def get_queries_and_context(self, context_tokens, input_info, ids_keep, ids_restore): # [6, 570, 256]
        B = context_tokens.shape[0]
        H, W = input_info["image_size"]
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        if "num_global_tokens" in input_info:
            context_tokens_without_global = context_tokens[:, : -input_info["num_global_tokens"]]
        else:
            context_tokens_without_global = context_tokens
        mask_tokens = repeat(self.mask_token, "() () d -> b n d", b=B, n=input_info["num_task_tokens"] - context_tokens_without_global.shape[1])
        context_with_mask = torch.cat([context_tokens_without_global, mask_tokens], dim=1)
        context_with_mask = torch.gather(context_with_mask, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))
        context_emb = self.generate_context_embeddings(input_info=input_info, bs=B, size=(N_H, N_W), device=context_tokens.device)
        context_with_mask = context_with_mask + context_emb
        if self.use_task_queries and self.task in input_info["tasks"]:
            start_idx = input_info["tasks"][self.task]["start_idx"]
            end_idx = input_info["tasks"][self.task]["end_idx"]
            queries = context_with_mask[:, start_idx:end_idx]
        else:
            queries = repeat(self.mask_token, "() () d -> b n d", b=B, n=N_H * N_W)
            queries_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bilinear", align_corners=False)
            queries_pos_emb = rearrange(queries_pos_emb, "b d nh nw -> b (nh nw) d")
            queries = queries + queries_pos_emb
            if self.task_embeddings is not None and self.task in self.task_embeddings:
                queries_task_emb = repeat(self.task_embeddings[self.task], "() () d -> b n d", b=B, n=N_H * N_W)
                queries = queries + queries_task_emb
        context_tokens_without_global = torch.gather(context_with_mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))
        if "num_global_tokens" in input_info:
            context_tokens = torch.cat([context_tokens_without_global, context_tokens[:, -input_info["num_global_tokens"] :]], dim=1)
        else:
            context_tokens = context_tokens_without_global
        return queries, context_tokens # [6, 1024, 256], [6, 570, 256]

    def forward(self, encoder_tokens: torch.Tensor, input_info: Dict, ids_keep: torch.Tensor, ids_restore: torch.Tensor): # [6, 570, 768]
        assert (self.dim_tokens_enc is not None), "Need to call init(dim_tokens_enc) function first"
        H, W = input_info["image_size"]
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        context_tokens = self.proj_context(encoder_tokens)
        queries, context_tokens = self.get_queries_and_context(context_tokens, input_info, ids_keep, ids_restore)
        if self.use_xattn:
            x = self.decoder(self.query_norm(queries), self.context_norm(context_tokens))
            x = x + self.mlp(self.out_norm(x))
        else:
            x = queries
        x = self.decoder_transformer(x)
        x = self.out_proj(x)
        x = rearrange(x, "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)", nh=N_H, nw=N_W, ph=self.P_H, pw=self.P_W, c=self.num_channels) # [6, 1, 512, 512]
        return x