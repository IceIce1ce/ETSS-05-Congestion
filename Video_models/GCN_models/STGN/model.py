import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.backbone1 = make_layers([64, 64, 'M', 128, 128])
        self.backbone2 = make_layers(['M', 256, 256, 256], in_channels=128)
        self.backbone3 = make_layers(['M', 512, 512, 512], in_channels=256)

    def forward(self, x): # [4, 3, 360, 640]
        x1 = self.backbone1(x) # [4, 128, 180, 320]
        x2 = self.backbone2(x1) # [4, 256, 90, 160]
        x3 = self.backbone3(x2) # [4, 512, 48, 80]
        return x1, x2, x3

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bn=False):
        super(BasicConv2d, self).__init__()
        self.use_bn = bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x): # [4, 128, 45, 80]
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x) # [4, 128, 45, 80]
        return x
    
class WCA(nn.Module):
    def __init__(self, channel):
        super(WCA, self).__init__()
        self.fc = nn.Sequential(BasicConv2d(channel, channel // 2, 1), nn.Conv2d(channel // 2, channel, 1))

    def forward(self, x): # [4, 384, 45, 80]
        b, c, _, _ = x.shape
        weight = x.reshape((b, c, -1))
        weight = torch.mean(weight * F.softmax(weight, dim=-1), dim=-1)
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        weight = F.softmax(self.fc(weight), dim=1)
        x = x * weight # [4, 384, 45, 80]
        return x

class Aggregation(nn.Module):
    def __init__(self, channel):
        super(Aggregation, self).__init__()
        self.channel = channel
        self.t1 = BasicConv2d(128, channel, 1)
        self.t2 = BasicConv2d(256, channel, 1)
        self.t3 = BasicConv2d(512, channel, 1)
        self.convs = nn.ModuleList([BasicConv2d(channel, channel, 1) for i in range(2)])
        self.conv_cats = nn.ModuleList([nn.Sequential(BasicConv2d(2*channel, channel, 1)) for i in range(2)])

    def forward(self, x1, x2, x3): # [4, 128, 180, 320], [4, 256, 90, 160], [4, 512, 45, 80]
        x1 = self.t1(x1)
        x2 = self.t2(x2)
        x3 = self.t3(x3)
        x1 = self.convs[0](F.interpolate(x1, size=x3.size()[2:], mode='bilinear', align_corners=True))
        x2 = self.convs[1](F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True))
        x2 = self.conv_cats[0](torch.cat((x2, x3), 1))
        x1 = self.conv_cats[1](torch.cat((x1, x2), 1)) # [4, 128, 45, 80]
        return x1

class MGL(nn.Module):
    def __init__(self, channel, dilation=1):
        super(MGL, self).__init__()
        self.fold = nn.Unfold(kernel_size=3, padding=dilation, dilation=dilation)
        self.conv = nn.ModuleList([BasicConv2d(channel, channel, 1) for _ in range(3)])

    def forward(self, x): # [1, 4, 128, 45, 80]
        n, t, c, h, w = x.shape
        x = x.view(n*t, c, h, w)
        x1 = self.conv[0](x)
        x2 = self.conv[1](x)
        x3 = self.conv[2](x)
        x1 = rearrange(self.fold(x1), '(n t) (c k2) hw -> n hw t k2 c', t=t, c=c)
        x1_var, x1_mean = torch.var_mean(x1, dim=3, unbiased=True)
        x1 = rearrange(x1, 'n hw t k2 c -> n hw (t k2) c')
        x2 = rearrange(self.fold(x2), '(n t) (c k2) hw -> n hw c t k2', t=t, c=c)
        x2_var, x2_mean = torch.var_mean(x2, dim=4, unbiased=True)
        x2 = rearrange(x2, 'n hw c t k2 -> n hw c (t k2)')
        score1 = F.softmax(torch.matmul(x1, x2), dim=-1)
        score2 = F.softmax(torch.matmul(x1_var, x2_var), dim=-1)
        score3 = F.softmax(torch.matmul(x1_mean, x2_mean), dim=-1)
        x3 = rearrange(self.fold(x3), '(n t) (c k2) hw -> n hw t (k2 c)', t=t, c=c)
        x4 = torch.matmul(score3, x3) + torch.matmul(score2, x3)
        x4 = rearrange(x4, 'n hw t (k2 c) -> n hw t k2 c', c=c)
        x3 = rearrange(x3, 'n hw t (k2 c) -> n hw (t k2) c', c=c)
        x5 = torch.matmul(score1, x3)
        x5 = rearrange(x5, 'n hw (t k2) c -> n hw t k2 c', t=t)
        x6 = x5 + x4
        kk = x6.shape[3]
        center = x6[:, :, :, kk//2, :].unsqueeze(-1)
        score4 = F.softmax(torch.matmul(x6, center), dim=-2)
        x = torch.sum(x6*score4, dim=-2)
        x = rearrange(x, 'n (h w) t c -> n t c h w', h=h) # [1, 4, 128, 45, 80]
        return x

class PGM(nn.Module):
    def __init__(self, channel, num=3):
        super(PGM, self).__init__()
        self.pool = nn.ModuleList([nn.AvgPool2d(kernel_size=2**i) for i in range(num)])
        self.conv = nn.ModuleList([BasicConv2d(channel, channel, 3, padding=1) for _ in range(num)])
        self.stgm = nn.ModuleList([MGL(channel) for i in range(num)])
        self.conv_cat = nn.Sequential(WCA(num*channel), BasicConv2d(num*channel, channel, 1), BasicConv2d(channel, channel, 3, padding=1))

    def forward(self, x): # [1, 4, 128, 45, 80]
        n, t, c, h, w = x.shape
        x = x.view(n*t, c, h, w)
        xs = []
        for pool, stgm, conv in zip(self.pool, self.stgm, self.conv):
            y = pool(x)
            _, _, h1, w1 = y.shape
            y = y.view(n, t, c, h1, w1)
            y = stgm(y)
            y = y.view(n*t, c, h1, w1)
            if h != h1:
                y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            y = conv(y)
            xs.append(y)
        x = self.conv_cat(torch.cat(xs, 1))
        x = x.view(n, t, c, h, w) # [1, 4, 128, 45, 80]
        return x

class STGN(nn.Module):
    def __init__(self, args):
        super(STGN, self).__init__()
        self.agg = args.agg
        self.backbone = BaseNet()
        if self.agg:
            self.aggregation = Aggregation(args.channel)
        else:
            self.aggregation = nn.Sequential(nn.Conv2d(512, args.channel, 1), nn.BatchNorm2d(args.channel), nn.ReLU())
        self.pgm = nn.ModuleList([PGM(args.channel) for _ in range(args.block_num)])
        self.conv_cat = nn.ModuleList([nn.Sequential(BasicConv2d((i + 1) * args.channel, args.channel, 1),
                                       BasicConv2d(args.channel, args.channel, 3, padding=1)) for i  in range(1, args.block_num + 1)])
        self.out = nn.Sequential(BasicConv2d(args.channel, args.channel, 3, padding=1), nn.Dropout2d(0.1), nn.Conv2d(args.channel, 1, 1))
        if self.training:
            self._initialize_weights()

    def forward(self, x): # [1, 4, 3, 360, 640]
        N, T, C, H, W = x.shape
        x = x.view(N * T, C, H, W)
        x1, x2, x3 = self.backbone(x)
        if self.agg:
            x = self.aggregation(x1, x2, x3)
        else:
            x = self.aggregation(x3)
        _, c, h, w = x.shape
        x = x.view(N, T, c, h, w)
        xs = []
        xs.append(x)
        for i, pgm in enumerate(self.pgm):
            x = pgm(x)
            xs.append(x)
            x = torch.cat(xs, dim=2)
            x = x.view(N*T, -1, h, w)
            x = self.conv_cat[i](x)
            x = x.view(N, T, c, h, w)
        x = x.view(N*T, c, h, w)
        x = self.out(x)
        x = x.view(N, T, -1, h, w) # [1, 4, 1, 45, 80]
        count = x.sum(dim=(2, 3, 4)) # [1, 4]
        return x, count
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        model = models.vgg16(True)
        my_models = self.backbone.state_dict()
        pre_models = model.state_dict()
        count = 0
        for layer_name, value in my_models.items():
            prelayer_name = list(pre_models.keys())[count]
            pre_weights = pre_models[prelayer_name]
            my_models[layer_name] = pre_weights
            count += 1
        self.backbone.load_state_dict(my_models)

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)