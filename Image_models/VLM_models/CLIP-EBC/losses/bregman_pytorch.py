# https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/bregman_pytorch.py
import torch
from torch import Tensor
from torch.cuda.amp import autocast
from typing import Union, Tuple, Dict

M_EPS = 1e-16

@autocast(enabled=True, dtype=torch.float32)
def sinkhorn(a: Tensor, b: Tensor, C: Tensor, reg: float = 1e-1, maxIter: int = 1000, stopThr: float = 1e-9, verbose: bool = False, log: bool = True, eval_freq: int = 10,
             print_freq: int = 200) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    device = a.device
    na, nb = C.shape
    assert na >= 1 and nb >= 1, f"C needs to be 2d. Found C.shape = {C.shape}"
    assert na == a.shape[0] and nb == b.shape[0], f"Shape of a ({a.shape}) or b ({b.shape}) does not match that of C ({C.shape})"
    assert reg > 0, f"reg should be greater than 0. Found reg = {reg}"
    assert a.min() >= 0. and b.min() >= 0., f"Elements in a and b should be nonnegative. Found a.min() = {a.min()}, b.min() = {b.min()}"
    if log:
        log = {"err": []}
    u = torch.ones((na), dtype=a.dtype).to(device) / na
    v = torch.ones((nb), dtype=b.dtype).to(device) / nb
    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)
    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)
    it = 1
    err = 1
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)
    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        KTu = torch.matmul(u.view(1, -1), K).view(-1)
        v = torch.div(b, KTu + M_EPS)
        Kv = torch.matmul(K, v.view(-1, 1)).view(-1)
        u = torch.div(a, Kv + M_EPS)
        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print("Warning: numerical errors at iteration", it)
            u, v = upre, vpre
            break
        if log and it % eval_freq == 0:
            b_hat = (torch.matmul(u.view(1, -1), K) * v.view(1, -1)).view(-1)
            err = (b - b_hat).pow(2).sum().item()
            log["err"].append(err)
        if verbose and it % print_freq == 0:
            print("iteration {:5d}, constraint error {:5e}".format(it, err))
        it += 1
    if log:
        log["u"] = u
        log["v"] = v
        log["alpha"] = reg * torch.log(u + M_EPS)
        log["beta"] = reg * torch.log(v + M_EPS)
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P