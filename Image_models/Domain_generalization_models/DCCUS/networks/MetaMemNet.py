from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
import math

def upsample_bilinear(x, size): # [32, 512, 14, 14], [32, 512, 28, 28]
    return F.interpolate(x, size=size[2:], mode='bilinear', align_corners=False) # [32, 512, 28, 28]

class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(Backbone, self).__init__()
        model = list(models.vgg16(pretrained=pretrained).features.children())
        self.feblock1 = nn.Sequential(*model[:16])
        self.feblock2 = nn.Sequential(*model[16:23])
        self.feblock3 = nn.Sequential(*model[23:30])
        self.beblock3 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.beblock2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.beblock1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x): # [32, 3, 224, 224]
        x = self.feblock1(x) # [32, 256, 56, 56]
        x1 = x
        x = self.feblock2(x) # [32, 512, 28, 28]
        x2 = x
        x = self.feblock3(x) # [32, 512, 14, 14]
        x = self.beblock3(x) # [32, 512, 14, 14]
        x3_ = x
        x = upsample_bilinear(x, x2.shape) # [32, 512, 28, 28]
        x = torch.cat([x, x2], 1) # [32, 1024, 28, 28]
        x = self.beblock2(x) # [32, 256, 28, 28]
        x2_ = x
        x = upsample_bilinear(x, x1.shape) # [32, 256, 56, 56]
        x = torch.cat([x, x1], 1) # [32, 512, 56, 56]
        x1_ = self.beblock1(x) # [32, 128, 56, 56]
        x2_ = upsample_bilinear(x2_, x1.shape) # [32, 256, 56, 56]
        x3_ = upsample_bilinear(x3_, x1.shape) # [32, 512, 56, 56]
        x = torch.cat([x1_, x2_, x3_], 1) # [32, 896, 56, 56]
        return x

class MetaMSNetBase(nn.Module):
    def __init__(self, pretrained=False):
        super(MetaMSNetBase, self).__init__()
        self.backbone = Backbone(True)
        self.output_layer = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        # self._initialize_weights()
        self.part_num = 1024
        variance = math.sqrt(1.0)
        self.sem_mem = nn.Parameter(torch.FloatTensor(1, 256, self.part_num).normal_(0.0, variance))
        self.sty_mem = nn.Parameter(torch.FloatTensor(4, 1, 256, self.part_num // 4).normal_(0.0, variance))
        self.sem_down = nn.Sequential(nn.Conv2d(512 + 256 + 128, 256, kernel_size=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))
        self.sty_down = nn.Sequential(nn.Conv2d(512 + 256 + 128, 256, kernel_size=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv_features(self, x): # [32, 3, 224, 224]
        x = self.backbone(x) # [32, 896, 56, 56]
        feature = self.sty_down(x) # [32, 256, 56, 56]
        return feature.unsqueeze(0) # [1, 32, 256, 56, 56]

    def train_forward(self, x, label): # [1, 3, 320, 320], 3
        size = x.shape
        x = self.backbone(x) # [1, 896, 80, 80]
        memory = self.sem_mem.repeat(x.shape[0], 1, 1) # [1, 256, 1024]
        memory_key = memory.transpose(1, 2) # [1, 1024, 256]
        sem_pre = self.sem_down(x) # [1, 256, 80, 80]
        sty_pre = self.sty_down(x) # [1, 256, 80, 80]
        sem_pre_ = sem_pre.view(sem_pre.shape[0], sem_pre.shape[1], -1) # [1, 256, 6400]
        diLogits = torch.bmm(memory_key, sem_pre_) # [1, 1024, 6400]
        invariant_feature = torch.bmm(memory_key.transpose(1, 2), F.softmax(diLogits, dim=1)) # [1, 256, 6400]
        recon_sim = torch.bmm(invariant_feature.transpose(1, 2), sem_pre_) # [1, 6400, 6400]
        sim_gt = torch.linspace(0, sem_pre.shape[2] * sem_pre.shape[3] - 1, sem_pre.shape[2] * sem_pre.shape[3]).unsqueeze(0).repeat(sem_pre.shape[0], 1).cuda() # [1, 6400]
        sim_loss = F.cross_entropy(recon_sim, sim_gt.long(), reduction='none') * 0.1
        invariant_feature_ = invariant_feature.view(invariant_feature.shape[0], invariant_feature.shape[1], sem_pre.shape[2], sem_pre.shape[3]) # [1, 256, 80, 80]
        den = self.output_layer(invariant_feature_) # [1, 1, 80, 80]
        den = upsample_bilinear(den, size=size) # [1, 1, 320, 320]
        memory2 = self.sty_mem[label].cuda() # [1, 256, 256]
        memory2 = memory2.repeat(x.shape[0], 1, 1) # [1, 256, 256]
        mem2_key = memory2.transpose(1, 2) # [1, 256, 256]
        sty_pre_ = sty_pre.view(sty_pre.shape[0], sty_pre.shape[1], -1) # [1, 256, 6400]
        dsLogits = torch.bmm(mem2_key, sty_pre_) # [1, 256, 6400]
        spe_feature = torch.bmm(mem2_key.transpose(1, 2), F.softmax(dsLogits, dim=1)) # [1, 256, 6400]
        recon_sim2 = torch.bmm(spe_feature.transpose(1, 2), sty_pre_) # [1, 6400, 6400]
        sim_gt2 = torch.linspace(0, sty_pre.shape[2] * sty_pre.shape[3] - 1, sty_pre.shape[2] * sty_pre.shape[3]).unsqueeze(0).repeat(sty_pre.shape[0], 1).cuda() # [1, 6400]
        sim_loss2 = F.cross_entropy(recon_sim2, sim_gt2.long(), reduction='sum') * 0.1
        orth_pre = torch.bmm(sty_pre_.transpose(1, 2), sem_pre_) # [1, 6400, 6400]
        orth_loss = 0.01 * torch.sum(torch.pow(torch.diagonal(orth_pre, dim1=-2, dim2=-1), 2))
        return den, sim_loss, sim_loss2, orth_loss

    def forward(self, x): # [1, 3, 1280, 1920]
        size = x.shape
        x = self.backbone(x) # [1, 896, 320, 480]
        memory = self.sem_mem.repeat(x.shape[0], 1, 1) # [1, 256, 1024]
        memory_key = memory.transpose(1, 2) # [1, 1024, 256]
        sem_pre = self.sem_down(x) # [1, 256, 320, 480]
        sem_pre_ = sem_pre.view(sem_pre.shape[0], sem_pre.shape[1], -1) # [1, 256, 153600]
        diLogits = torch.bmm(memory_key, sem_pre_) # [1, 1024, 153600]
        invariant_feature = torch.bmm(memory_key.transpose(1, 2), F.softmax(diLogits, dim=1)) # [1, 256, 153600]
        invariant_feature = invariant_feature.view(invariant_feature.shape[0], invariant_feature.shape[1], sem_pre.shape[2], sem_pre.shape[3]) # [1, 256, 320, 480]
        den = self.output_layer(invariant_feature) # [1, 1, 320, 480]
        den = upsample_bilinear(den, size=size) # [1, 1, 1280, 1920]
        return den

class MetaMemNet(nn.Module):
    def getBase(self):
        baseModel = MetaMSNetBase(True)
        return baseModel

    def __init__(self):
        super(MetaMemNet, self).__init__()
        self.base = self.getBase()

    def train_forward(self, x, label):
        dens, sim_loss, sim_loss2, orth_loss = self.base.train_forward(x, label) # [1, 1, 320, 320]
        dens = upsample_bilinear(dens, x.shape) # [1, 1, 320, 320]
        return dens, sim_loss, sim_loss2, orth_loss

    def forward(self, x): # [1, 3, 1280, 1920]
        dens = self.base(x) # [1, 1, 1280, 1920]
        dens = upsample_bilinear(dens, x.shape) # [1, 1, 1280, 1920]
        return dens

    def get_grads(self):
        grads = []
        for p in self.base.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.base.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def conv_features(self, x): # [32, 3, 224, 224]
        x = self.base.conv_features(x) # [1, 32, 256, 56, 56]
        return x