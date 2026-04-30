import torch
import torch.nn as nn
from config import cfg
from .resSFCN import resSFCN as net

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()        
        self.CCN = net()
        if len(cfg.TRAIN.GPU_ID) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=cfg.TRAIN.GPU_ID).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map): # [2, 3, 576, 768], [2, 576, 768]
        density_map = self.CCN(img).squeeze() # [2, 576, 768]
        self.loss_mse = self.build_loss(density_map, gt_map)
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map