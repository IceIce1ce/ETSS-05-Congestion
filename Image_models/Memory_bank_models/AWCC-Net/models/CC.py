import torch.nn as nn
import torch
from models.AWCC import vgg19_trans

class CrowdCounter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = vgg19_trans(use_pe=True)

    @torch.no_grad()
    def test_forward(self, x, **kwargs): # [1, 3, 955, 1300]
        input_list = [x]
        dmap_list = []
        for input in input_list:
            pred_map, _ = self.net(input) # [1, 1, 59, 81]
            dmap_list.append(pred_map.detach())
        return torch.relu(dmap_list[0]) # [1, 1, 59, 81]