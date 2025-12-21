import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from .conv import ResBlock
from model.necks import FPN

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

class ResNet_50_FPN_Encoder(nn.Module):
    def __init__(self):
        super(ResNet_50_FPN_Encoder, self).__init__()
        resnet_50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = resnet_50.conv1
        self.bn1 = resnet_50.bn1
        self.relu = resnet_50.relu
        self.maxpool = resnet_50.maxpool
        self.layer1 = resnet_50.layer1
        self.layer2 = resnet_50.layer2
        self.layer3 = resnet_50.layer3
        self.layer4 = resnet_50.layer4
        in_channels = [512, 1024, 2048]
        self.neck2f = FPN(in_channels, 256, len(in_channels))
        self.feature_head = nn.Sequential(nn.Dropout2d(0.2), ResBlock(in_dim=768, out_dim=384, dilation=0, norm="bn"), ResBlock(in_dim=384, out_dim=256, dilation=0, norm="bn"),
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), BatchNorm2d(256, momentum=BN_MOMENTUM),
                                          nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f_list = []
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        f_list.append(x2)
        x3 = self.layer3(x2)
        f_list.append(x3)
        x4 = self.layer4(x3)
        f_list.append(x4)

        fpn_f_list = self.neck2f(f_list)
        outputs = []
        outputs.append(F.interpolate(fpn_f_list[0],scale_factor=0.5, mode='bilinear', align_corners=True))
        outputs.append(fpn_f_list[1])
        outputs.append(F.interpolate(fpn_f_list[2],scale_factor=2, mode='bilinear', align_corners=True))
        multi_scale_f = torch.cat([outputs[0], outputs[1], outputs[2]], dim=1)
        feature = self.feature_head(multi_scale_f)
        outputs.append(feature)
        return outputs
