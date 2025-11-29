import torch.nn as nn
import torch.nn.functional as F

conv_cfg = {'Conv': nn.Conv2d}

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride=1, padding=0,dilation=1, norm=None, relu =False):
        super(BasicConv, self).__init__()
        self.relu = relu
        bias = True if  norm is None else  False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding,dilation=dilation, bias=bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.01)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == None:
            self.norm = None

    def forward(self, x): # [4, 192, 192, 256]
        x = self.conv(x)
        x = self.norm(x) if self.norm is  not None else x
        x = F.relu(x, inplace=True) if self.relu else x # [4, 48, 192, 256]
        return x

class ResBlock(nn.Module):
    def __init__(self, in_dim,out_dim, dilation=1,  norm="bn"):
        super(ResBlock, self).__init__()
        padding = dilation + 1
        model = []
        medium_dim = in_dim // 4
        model.append(BasicConv(in_dim, medium_dim, 1, 1, 0, norm = norm, relu =True))
        model.append(BasicConv(medium_dim, medium_dim, 3, 1, padding = padding, dilation=dilation+1, norm=norm,  relu =True))
        model.append(BasicConv(medium_dim, out_dim, 1, 1, 0, norm=norm, relu =False))
        self.model = nn.Sequential(*model)
        if in_dim !=out_dim:
            self.downsample =  BasicConv(in_dim, out_dim, 1, 1, 0, norm=norm, relu =False)
        else:
            self.downsample =None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): # [4, 192, 192, 256]
        residual = x
        out = self.model(x)
        if self.downsample  is not None:
            out = out + self.downsample(residual)
        else:
            out = out + residual
        out = self.relu(out) # [4, 128, 192, 256]
        return out