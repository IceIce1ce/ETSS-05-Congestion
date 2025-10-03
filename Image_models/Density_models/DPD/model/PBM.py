import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

class BinarizedF(Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input,threshold)
        a = torch.ones_like(input).cuda()
        b = torch.zeros_like(input).cuda()
        output = torch.where(input>=threshold,a,b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight  = None
        if ctx.needs_input_grad[0]:
          grad_input= 0.2 * grad_output
        if ctx.needs_input_grad[1]:
          grad_weight = -grad_output
        return grad_input, grad_weight

class compressedSigmoid(nn.Module):
    def __init__(self, para=2.0, bias=0.2):
        super(compressedSigmoid, self).__init__()
        self.para = para
        self.bias = bias

    def forward(self, x): # [6, 1, 64, 128]
        output = 1. / (self.para + torch.exp(-x)) + self.bias # [6, 1, 64, 128]
        return output

class BinarizedModule(nn.Module):
    def __init__(self, input_channels=720):
        super(BinarizedModule, self).__init__()
        self.Threshold_Module = nn.Sequential(nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(),
                                              nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(),
                                              nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(),
                                              nn.AvgPool2d(15, stride=1, padding=7), nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
                                              nn.AvgPool2d(15, stride=1, padding=7))
        self.sig = compressedSigmoid()

    def forward(self, feature, pred_map): # [6, 720, 128, 256], [6, 1, 512, 1024]
        p = F.interpolate(pred_map.detach(), scale_factor=0.125)
        f = F.interpolate(feature.detach(), scale_factor=0.5)
        f = f * p
        threshold = self.Threshold_Module(f)
        threshold = self.sig(threshold * 10.)
        threshold = F.interpolate(threshold, scale_factor=8) # [6, 1, 512, 1024]
        Binar_map = BinarizedF.apply(pred_map, threshold) # [6, 1, 512, 1024]
        return threshold, Binar_map