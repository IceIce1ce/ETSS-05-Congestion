import torch
import torch.nn as nn
from torchvision import models
from models.FourierEncoding import PositionalEncoding

class New_bay_Net(nn.Module):
    def __init__(self, input_size):
        super(New_bay_Net, self).__init__()
        self.input_size = input_size
        self.ASRNet = ASRNet()
        self.m = 64
        self.z_feature_size = 16
        self.pos_encode_layer = PositionalEncoding(0.5, self.m)
        pos_out_dim = 2 * 2 * self.m
        weight_dim = pos_out_dim + 3*self.z_feature_size*self.z_feature_size + 4
        self.Encoder2z = Encoder2z(self.input_size, weight_dim, self.z_feature_size)
        self.cc_decoder = build_cc_decoder(self.z_feature_size, self.m, pos_out_dim)
        self.kl_div = 0

    def forward(self, x, grid_c, mode): # [16, 3, 256, 256], [16, 1024, 2]. train
        x = self.ASRNet(x) # [16, 512, 32, 32]
        grid_c = self.pos_encode_layer(grid_c) # [16, 1024, 256]
        x = self.Encoder2z(x) # [16, 1028, 16, 16]
        if mode == 'train':
            mod = True
        else:
            mod = False
        x, out_sigma, kl_div = self.cc_decoder(x, grid_c, mod) # [16, 1024], [16, 1024]
        self.kl_div = kl_div
        return x, out_sigma

class ASRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(ASRNet, self).__init__()
        self.resnet_backbone = Res_Backbone()
        self.output_layer = nn.Conv2d(512, 512, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()       
            for i in range(len(self.resnet_backbone.state_dict().items())):
                list(self.resnet_backbone.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x): # [16, 3, 256, 256]
        x, f1, f2, f3 = self.resnet_backbone(x) # [16, 512, 32, 32], [16, 64, 256, 256], [16, 128, 128, 128], [16, 256, 64, 64]
        x = self.output_layer(x) # [16, 512, 32, 32]
        return torch.abs(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Res_Backbone(nn.Module):
    def __init__(self, load_weights=False):
        super(Res_Backbone, self).__init__()
        self.frontend_feat1 = [64, 64]
        self.frontend_feat2 = ['M', 128, 128]
        self.frontend_feat3 = ['M', 256, 256, 256]
        self.frontend_feat4 = ['M', 512, 512, 512]
        self.frontend1 = make_layers_2(self.frontend_feat1)
        self.frontend2 = make_layers_2(self.frontend_feat2, in_channels = 64)
        self.frontend3 = make_layers_2(self.frontend_feat3, in_channels = 128)
        self.frontend4 = make_layers_2(self.frontend_feat4, in_channels = 256)

    def forward(self, x):
        x = self.frontend1(x)
        f1 = x
        x = self.frontend2(x)
        f2 = x
        x = self.frontend3(x)
        f3 = x
        x = self.frontend4(x)
        return x, f1, f2, f3

class Encoder2z(nn.Module):
    def __init__(self, input_size, m_size, z_feature_size, load_weights=False):
        super(Encoder2z, self).__init__()
        self.m_size = m_size
        self.ratio = 1
        self.frontend_feat3 = []
        self.frontend_feat4 = []
        for i in range(self.ratio):
            self.frontend_feat3 += ['M',512]
        self.frontend_feat4 += [512, self.m_size]
        self.frontend3 = make_layers_4(self.frontend_feat3, in_channels = 512, batch_norm = True)
        self.frontend4 = make_layers_4(self.frontend_feat4, in_channels = 512, batch_norm = True)
        self.output_layer = nn.Conv2d(self.m_size, self.m_size, kernel_size=1)
        self._initialize_weights() 
        
    def forward(self, x):
        x = self.frontend3(x)
        x = self.frontend4(x)
        x = self.output_layer(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.001)

class CC_Decoder(nn.Module):
    def __init__(self, feature_size, m_size, pos_out_dim):
        super(CC_Decoder, self).__init__()
        self.n_features = int(feature_size*feature_size)
        self.pos_dim = pos_out_dim
        self.weight_dim = self.pos_dim + 3 * self.n_features + 4
        self.last1 = nn.Linear(self.n_features, 1)
        self.last2 = torch.nn.Sequential(nn.Linear(self.n_features, self.n_features), nn.PReLU(),
                                         nn.Linear(self.n_features, 1))
        self.act = nn.PReLU()
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act4 = nn.PReLU()
        self.W_fine = nn.Linear(self.n_features, self.n_features)
        self._initialize_weights()
        self.kl = 0

    def forward(self, x1, x2, mode): # [16, 1028, 16, 16], [16, 1024, 256], True
        b, n_query_pts = x2.shape[0], x2.shape[1]
        W = torch.reshape(x1, (b, self.weight_dim, self.n_features))
        W = self.W_fine(W)
        W1 = W[:,:self.pos_dim,:]
        b1 = W[:,self.pos_dim:self.pos_dim+1,:].repeat(1, n_query_pts, 1) / 10
        W2 = W[:,(self.pos_dim+1):(self.pos_dim+self.n_features+1),:]
        b2 = W[:,(self.pos_dim+self.n_features+1):(self.pos_dim+self.n_features+2),:].repeat(1, n_query_pts, 1)/10
        W3 = W[:,(self.pos_dim+self.n_features+2):(self.pos_dim+2*self.n_features+2),:]
        b3 = W[:,(self.pos_dim+2*self.n_features+2):(self.pos_dim+2*self.n_features+3),:].repeat(1, n_query_pts, 1)/10
        W4 = W[:,(self.pos_dim+2*self.n_features+3):(self.pos_dim+3*self.n_features+3),:]
        b4 = W[:,(self.pos_dim+3*self.n_features+3):(self.pos_dim+3*self.n_features+4),:].repeat(1, n_query_pts, 1)/10
        out1 = torch.einsum("bij, bjk -> bik", x2, W1) + b1
        out1 = self.act1(out1)
        out2 = torch.einsum("bij, bjk -> bik", out1, W2) + b2
        out2 = self.act2(out2) + out1
        out3 = torch.einsum("bij, bjk -> bik", out2, W3) + b3
        out3 = self.act3(out3) + out2
        out4 = torch.einsum("bij, bjk -> bik", out3, W4) + b4
        out4 = self.act4(out4) + out3
        out_mu = torch.squeeze(self.act(self.last1(out4)))
        out_sigma = torch.exp(torch.squeeze(self.last2(out4)))
        out = out_mu
        self.kl = torch.mean((0.5 * out_sigma**2 + 0.5 * (out_mu)**2 - torch.log(out_sigma) - 1 / 2))
        return out, out_sigma, self.kl # [16, 1024], [16, 1024]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def build_cc_decoder(feature_size, m_size, pos_out_dim):
    return CC_Decoder(feature_size, m_size, pos_out_dim)

def make_layers_2(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_4(cfg, in_channels = 3,batch_norm=False,dilation = False):
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
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU()]
            else:
                layers += [conv2d, nn.PReLU()]
            in_channels = v
    return nn.Sequential(*layers)