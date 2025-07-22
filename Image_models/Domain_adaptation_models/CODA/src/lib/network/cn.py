import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class CNet(nn.Module):
    def __init__(self, in_place):
        super(CNet, self).__init__()
        self.in_place = in_place
        self.fc6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.fc7 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=4, dilation=4)
        self.densitymap = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear')

    def forward(self, x): # [16, 3, 512, 512]
        for i in range(len(self.CNN_base)):
            x = self.CNN_base[i](x)
            if i == 22:
                x4_3 = x
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.densitymap(x)
        x = self.upsample(x) / 16 # [16, 1, 512, 512]
        return x

    def _init_weights(self,truncate=False):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.fc6, 0, 0.01, truncate)
        normal_init(self.fc7, 0, 0.01, truncate)
        normal_init(self.densitymap, 0, 0.01, truncate)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

class vgg16(CNet):
    name = 'CNet_VGG'
    def __init__(self, pretrained=True):
        self.model_path = 'pretrained_models/vgg16_caffe.pth'
        self.pretrained = pretrained
        CNet.__init__(self, in_place=256)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print('Loading ckpt from: %s' % self.model_path)
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
        self.CNN_base = nn.Sequential(*list(vgg.features._modules.values())[:30])
        for layer in range(10):
            for p in self.CNN_base[layer].parameters():
                p.requires_grad = False