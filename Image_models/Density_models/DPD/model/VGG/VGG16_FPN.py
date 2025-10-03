from torchvision import models
import torch.nn.functional as F
from misc.utils import initialize_weights
import torch.nn as nn
import torch

mode = 'Vgg_bn'

class VGG16_FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_FPN, self).__init__()
        if mode == 'Vgg_bn':
            vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        if  mode == 'Vgg_bn':
            self.layer1 = nn.Sequential(*features[0:23])
            self.layer2 = nn.Sequential(*features[23:33])
            self.layer3 = nn.Sequential(*features[33:43])
        in_channels = [256, 512, 512]
        self.neck =  FPN(in_channels, 256, len(in_channels))
        self.de_pred = nn.Sequential(nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(768), nn.ReLU(inplace=False),
                                     nn.ConvTranspose2d(768, 64, 4, stride=2, padding=1, output_padding=0, bias=True), nn.ReLU(inplace=False),
                                     nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=True), nn.Sigmoid())
        initialize_weights(self.de_pred)

    def forward(self, x): # [6, 3, 512, 1024]
        f = []
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        f = self.neck(f)
        feature = torch.cat([f[0], F.interpolate(f[1], scale_factor=2), F.interpolate(f[2], scale_factor=4)], dim=1) # [6, 768, 128, 256]
        x = self.de_pred(feature) # [6, 1, 512, 1024]
        return feature, x

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, bn=True):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2d( in_channels[i], out_channels,1,bn=bn, bias=not bn,same_padding=True)
            fpn_conv = Conv2d( out_channels, out_channels,3,bn=bn, bias=not bn,same_padding=True)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = F.interpolate(laterals[i], size=prev_shape, mode='nearest') + laterals[i - 1]
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=False)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x