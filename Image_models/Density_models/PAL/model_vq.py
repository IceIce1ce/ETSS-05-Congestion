# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L216
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch import distributed as dist

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5): # 64, 512
        super(Quantize, self).__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, mask=None, train_flag=True): # [1, 64, 64, 64], None, False
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(flatten, self.embed)  + self.embed.pow(2).sum(0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        if mask is not None:
            input_new = input[mask.bool()] # [578, 64]
            flatten_new = input_new
            dist_new = flatten_new.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(flatten_new, self.embed) + self.embed.pow(2).sum(0, keepdim=True)
            _, embed_ind_new = (-dist_new).max(1)
            embed_onehot_new = F.one_hot(embed_ind_new,self.n_embed).type(flatten_new.dtype) # [578, 512]
        if self.training and train_flag:
            embed_onehot_sum = embed_onehot_new.sum(0) # [512]
            embed_sum = torch.matmul(flatten_new.transpose(0, 1), embed_onehot_new)
            all_reduce(embed_onehot_sum)
            all_reduce(embed_sum)
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0) # [64, 512]
            self.embed.data.copy_(embed_normalized)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind # [1, 64, 64, 64], [1], [1, 64, 64]

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))