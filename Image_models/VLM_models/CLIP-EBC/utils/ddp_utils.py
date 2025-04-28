import torch.distributed as dist
from torch import Tensor

def reduce_mean(tensor: Tensor, nprocs: int) -> Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def cleanup(ddp: bool = True) -> None:
    if ddp:
        dist.destroy_process_group()

def barrier(ddp: bool = True) -> None:
    if ddp:
        dist.barrier()