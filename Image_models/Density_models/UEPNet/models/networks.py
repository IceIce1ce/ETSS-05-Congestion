import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=(), do_init=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if do_init:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def vgg_make_layers(cfg, in_channels = 3, batch_norm = False, dilation = False):
    if dilation: 
        d_rate = 2
    else: 
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = d_rate, dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)   

class VGGNet(nn.Module):
    def __init__(self, batch_norm=False, post_pool=False):
        super(VGGNet, self).__init__()
        if not post_pool:
            self.frontend_1_feat = [64, 64]
            self.frontend_2_feat = ['M', 128, 128]
            self.frontend_3_feat = ['M', 256, 256, 256]
            self.frontend_4_feat = ['M', 512, 512, 512]
            self.frontend_5_feat = ['M', 512, 512, 512]
        else:
            self.frontend_1_feat = [64, 64, 'M']
            self.frontend_2_feat = [128, 128, 'M']
            self.frontend_3_feat = [256, 256, 256, 'M']
            self.frontend_4_feat = [512, 512, 512, 'M']
            self.frontend_5_feat = [512, 512, 512, 'M']
        self.frontend_1 = vgg_make_layers(self.frontend_1_feat, in_channels=3, batch_norm=batch_norm)
        self.frontend_2 = vgg_make_layers(self.frontend_2_feat, in_channels=64, batch_norm=batch_norm)
        self.frontend_3 = vgg_make_layers(self.frontend_3_feat, in_channels=128, batch_norm=batch_norm)
        self.frontend_4 = vgg_make_layers(self.frontend_4_feat, in_channels=256, batch_norm=batch_norm)
        self.frontend_5 = vgg_make_layers(self.frontend_5_feat, in_channels=512, batch_norm=batch_norm)

    def forward(self,x):
        e1 = self.frontend_1(x)
        e2 = self.frontend_2(e1)
        e3 = self.frontend_3(e2)
        e4 = self.frontend_4(e3)
        e5 = self.frontend_5(e4)
        return e1, e2, e3, e4, e5