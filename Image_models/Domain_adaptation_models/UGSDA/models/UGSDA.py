import torch.nn as nn
from .SASNet_own import SASNet_own as ccnet

class UGSDA(nn.Module):
    def __init__(self):
        super(UGSDA, self).__init__()
        self.CCN = ccnet()
    
    def forward(self, img):
        pass

    def test_forward(self, img): # [1, 3, 1280, 1920]
        density_map = self.CCN(img) # [1, 1, 1280, 1920]
        return density_map