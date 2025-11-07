import torch
import torch.nn as nn
from .HRNet import HighResolutionNet as net

class ObjectCounter(nn.Module):
    def __init__(self, gpus):
        super(ObjectCounter, self).__init__()
        self.CCN = net()
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()

    def test_forward(self, img): # [1, 3, 720, 1280]
        density_map = self.CCN(img) # [1, 1, 720, 1280]
        return density_map