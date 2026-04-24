import torch
import torch.nn as nn
from torchvision import models
from model_vq import Quantize
from model_csrnet import CSRNet as net_csr

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)
        # VQ config
        embed_dim = 64
        n_embed = 512
        self.quantize = Quantize(embed_dim, n_embed)
        channel = 512
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_deconv = nn.Conv2d(embed_dim, channel, 1)
        # CSRNet config
        self.csrnet_1 = net_csr()
        self.csrnet_2 = net_csr()
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for (frontend_key, frontend_val), (mod_key, mod_val) in zip(self.frontend.state_dict().items(), mod.state_dict().items()):
                frontend_val.data[:] = mod_val.data[:]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x, mask=None, target=None, train_flag=True): # [1, 3, 341, 512], None, None, False
        diff_mean, diff_var = None, None
        x = self.frontend(x) # [1, 512, 42, 64]
        x_dis = x.clone() # [1, 512, 42, 64]
        if target is not None:
            target = (target - target.min() )/ (target.max() - target.min()) # [1, 1, 23, 35]
            x_dis = x_dis * target  #[1, 512, 23, 35]
        if mask is not None:
            diff_mean, diff_var=[], []
            x_dis = x_dis.permute(0, 2, 3, 1) # [1, 48, 64, 512]
            x_ann = x_dis[mask.bool()] # [646, 512]
            x_unkn = x_dis[mask.bool()==0] # [2426, 512]
            x_ann_mean, x_ann_var = torch.var_mean(x_ann, dim=0) # [512], [512]
            x_unkn_mean, x_unkn_var = torch.var_mean(x_unkn, dim=0) # [512], [512]
            x_ann_ = x_ann - x_ann_mean # [850, 512]
            x_unkn_ = x_unkn - x_unkn_mean # [1692, 512]
            x_ann_var = torch.matmul(x_ann_.permute(1, 0), x_ann_) / x_ann_.shape[0] # [512, 512]
            x_unkn_var = torch.matmul( x_unkn_.permute(1,0), x_unkn_ ) / x_unkn_.shape[0] # [512, 512]
            diff_mean.append(x_ann_mean)
            diff_mean.append(x_unkn_mean)
            diff_var.append(x_ann_var)
            diff_var.append(x_unkn_var)
        quant = self.quantize_conv(x).permute(0, 2, 3, 1) # [1, 29, 23, 64]
        quant, diff, id_t = self.quantize(quant, mask, train_flag) # [1, 29, 23, 64], [1], [1, 29, 23]
        quant = quant.permute(0, 3, 1, 2) # [1, 64, 29, 23]
        quant = self.quantize_deconv(quant) # [1, 512, 29, 23]
        x = torch.cat([x, quant], dim=1) # [1, 1024, 29, 23]
        output_csrnet_1 = self.csrnet_1(x) # [1, 1, 29, 23]
        output_csrnet_2 = self.csrnet_2(x) # [1, 1, 29, 23]
        return output_csrnet_1, output_csrnet_2, diff, id_t, diff_mean, diff_var

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)