import torch
import numpy as np
import torch.nn as nn
from torch import Tensor

@torch.jit.script
def positional_encoding(v: Tensor, sigma: float, m: int) -> Tensor:
    j = torch.arange(m, device=v.device)
    coeffs = 2.0** j * np.pi ###2.0
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, sigma: float, m: int):
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: Tensor) -> Tensor:
        return positional_encoding(v, self.sigma, self.m)