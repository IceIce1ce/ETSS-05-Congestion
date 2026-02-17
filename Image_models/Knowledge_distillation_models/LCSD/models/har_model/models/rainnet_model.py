import torch
from .base_model import BaseModel
from . import networks
from torch import nn

class RainNetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_gp', 'D_global', 'D_local', 'G_global', 'G_local']
        self.visual_names = ['comp', 'real', 'output', 'mask', 'real_f', 'fake_f', 'bg', 'attentioned']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()
        if self.isTrain: 
            self.gan_mode = opt.gan_mode
            netD = networks.NLayerDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, networks.get_norm_layer(opt.normD))
            self.netD = networks.init_net(netD, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*opt.d_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.iter_cnt = 0

    def set_input(self, input):
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.inputs = self.comp
        if self.opt.input_nc == 4:
            self.inputs = torch.cat([self.inputs, self.mask], 1)
        self.real_f = self.real * self.mask
        self.bg = self.real * (1 - self.mask)

    def forward(self):
        self.output = self.netG(self.inputs, self.mask)
        self.fake_f = self.output * self.mask
        self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
        self.harmonized = self.attentioned

    def backward_D(self):
        fake_AB = self.harmonized
        pred_fake, ver_fake = self.netD(fake_AB.detach(), self.mask)
        if self.gan_mode == 'wgangp':
            global_fake = self.relu(1 + pred_fake).mean()
            local_fake = self.relu(1 + ver_fake).mean()
        else:
            global_fake = self.criterionGAN(pred_fake, False)
            local_fake = self.criterionGAN(ver_fake, False)
        self.loss_D_fake = global_fake + local_fake
        real_AB = self.real
        pred_real, ver_real = self.netD(real_AB, self.mask)
        if self.gan_mode == 'wgangp':
            global_real = self.relu(1 - pred_real).mean()
            local_real = self.relu(1 - ver_real).mean()
        else:
            global_real = self.criterionGAN(pred_real, True)
            local_real = self.criterionGAN(ver_real, True)
        self.loss_D_real = global_real + local_real
        self.loss_D_global = global_fake + global_real
        self.loss_D_local = local_fake + local_real
        gradient_penalty, gradients = networks.cal_gradient_penalty(self.netD, real_AB.detach(), fake_AB.detach(), 'cuda', mask=self.mask)
        self.loss_D_gp = gradient_penalty
        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.opt.gp_ratio * gradient_penalty)
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        fake_AB = self.harmonized
        pred_fake, ver_fake, featg_fake, featl_fake = self.netD(fake_AB, self.mask, feat_loss=True)
        self.loss_G_global = self.criterionGAN(pred_fake, True)
        self.loss_G_local = self.criterionGAN(ver_fake, True)
        self.loss_G_GAN =self.opt.lambda_a * self.loss_G_global + self.opt.lambda_v * self.loss_G_local
        self.loss_G_L1 = self.criterionL1(self.attentioned, self.real) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
