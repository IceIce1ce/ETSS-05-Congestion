import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel
from . import networks
from .networks import VGGNet

class DoubleHead(nn.Module):
    def __init__(self, num_classes):
        super(DoubleHead, self).__init__()
        self.PredBackbone1 = nn.Sequential(nn.Conv2d(512, 512, (1, 1)), nn.ReLU()) 
        self.PredBackbone2 = nn.Sequential(nn.Conv2d(512, 512, (1, 1)), nn.ReLU()) 
        self.head1 = nn.Conv2d(512, num_classes, (1, 1))
        self.head2 = nn.Conv2d(512, num_classes + 1, (1, 1))

    def forward(self, x):
        x1 = self.PredBackbone1(x)
        x2 = self.PredBackbone2(x)
        return self.head1(x1), self.head2(x2)

class UEPModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--with_bn', action='store_true')
        parser.add_argument('--freeze_bn', action='store_true')
        parser.add_argument('--vgg_post_pool', action='store_true')
        parser.add_argument('--heatmap_multi', default=1.0, type=float)
        parser.add_argument('--psize', default=8, type=int)
        parser.add_argument('--pstride', default=8, type=int)
        parser.add_argument('--folder', default=1, type=int)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.model_names = ['VggBackbone', 'DivideUpsample0', 'DivideUpsample1', 'ClsPred']
        if self.opt.dataset_mode == 'shtechparta':
            self.label_indices = np.array([0.00016, 0.0048202634789049625, 0.01209819596260786, 0.02164922095835209, 0.03357841819524765, 0.04810526967048645, 0.06570728123188019, 0.08683456480503082, 0.11207923293113708, 0.1422334909439087, 0.17838051915168762,
                                           0.22167329490184784, 0.2732916474342346, 0.33556100726127625, 0.41080838441848755, 0.5030269622802734, 0.6174761652946472, 0.762194037437439, 0.9506691694259644, 1.2056223154067993, 1.5706151723861694, 2.138580322265625,
                                           3.233219861984253, 7.914860725402832])
            self.indices_proxy = np.array([0, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624, 0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904, 0.15925543689080027,
                                           0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514, 0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761, 1.3645639664610938, 1.8055319029995607,
                                           2.541316177212592, 3.87642023839676, 8.247815291086832])
            self.indices_proxy2 = np.array([0, 0.0008736941759623788, 0.00460105649110827, 0.011909992029514994, 0.021447560775165905, 0.03335742127399603, 0.04785158393927123, 0.06538952954794941, 0.08647975537451662, 0.11168024780931907, 0.14175821026385504,
                                            0.17778540202168958, 0.22097960677712483, 0.2724192081348686, 0.3344926685808885, 0.40938709885499597, 0.5012436541947841, 0.6149288298909453, 0.7585325340575756, 0.9452185066011628, 1.1967563985336944, 1.5541906336372862,
                                            2.0969205546489382, 2.9970217618726727, 4.51882041862729, 8.527834415435791])
        self.num_class = self.label_indices.size + 1
        self.netVggBackbone = VGGNet(self.opt.with_bn, self.opt.vgg_post_pool)
        self.netDivideUpsample0 = Upsample(512, 256, 256 + 512, 512)
        self.netDivideUpsample1 = Upsample(512, 256, 256 + 256, 512)
        self.netClsPred = DoubleHead(self.num_class)
        self.netVggBackbone = networks.init_net(self.netVggBackbone, gpu_ids=self.gpu_ids, do_init=False)
        self.netDivideUpsample0 = networks.init_net(self.netDivideUpsample0, init_type='normal', init_gain=0.01, gpu_ids=self.gpu_ids)
        self.netDivideUpsample1 = networks.init_net(self.netDivideUpsample1, init_type='normal', init_gain=0.01, gpu_ids=self.gpu_ids)
        self.netClsPred = networks.init_net(self.netClsPred, init_type='normal', init_gain=0.01, gpu_ids=self.gpu_ids)

    def to_eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def set_input(self, input):
        self.data_img = input['img'].to(self.device) 
        self.data_gt = input['gt'].to(self.device) if 'gt' in input else None

    def forward(self):
        conv1, conv2, conv3, conv4, conv5 = self.netVggBackbone(self.data_img)
        feat0 = conv5
        feat1 = self.netDivideUpsample0(feat0, conv4)
        feat2 = self.netDivideUpsample1(feat1, conv3)
        self.cls, self.cls2 = self.netClsPred(feat2)
        self.count1 = class_to_count(self.cls.max(dim=1, keepdim=True)[1], self.indices_proxy)
        self.count2 = class_to_count(self.cls2.max(dim=1, keepdim=True)[1], self.indices_proxy2)
        self.count = (self.count1 + self.count2) / 2.
        self.pred_heatmap = self.count / self.opt.heatmap_multi

    def predict(self, return_sum=True):
        self.forward()
        if return_sum:
            if self.pred_heatmap.device.type == 'cuda':
                return self.pred_heatmap.data.cpu().numpy().sum()
            return self.pred_heatmap.data.numpy().sum()
        else:
            if self.pred_heatmap.device.type == 'cuda':
                return self.pred_heatmap.data.cpu().numpy()
            return self.pred_heatmap.data.numpy()

class Upsample(nn.Module):
    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(*[nn.Conv2d(up_in_ch, up_out_ch, 3, 1, padding=1), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv2d(cat_in_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU(), nn.Conv2d(cat_out_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])
        
    def forward(self, low, high):
        low = self.up(low)
        low = self.conv1(low)
        x = torch.cat([high, low], dim=1)
        x = self.conv2(x)
        return x

def class_to_count(pre_cls, indices_proxy):
    on_gpu = (pre_cls.device.type == 'cuda')
    label2count = torch.tensor(indices_proxy)
    label2count = label2count.type(torch.FloatTensor)
    input_size = pre_cls.size()
    pre_cls = pre_cls.reshape(-1).cpu()
    pre_counts = torch.index_select(label2count, 0, pre_cls.cpu().type(torch.LongTensor))
    pre_counts = pre_counts.reshape(input_size)
    if on_gpu:
        pre_counts = pre_counts.cuda()
    return pre_counts