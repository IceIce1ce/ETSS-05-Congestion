import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def upsample_bilinear(x, size):
	return F.interpolate(x, size=size[2:], mode='bilinear', align_corners=False)

class SASNet_own(nn.Module):
	def __init__(self):
		super(SASNet_own, self).__init__()
		model = list(models.vgg16(pretrained=True).features.children())
		self.feblock = nn.Sequential(*model[:4])
		self.feblock1 = nn.Sequential(*model[4:16])
		self.feblock2 = nn.Sequential(*model[16:23])
		self.feblock3 = nn.Sequential(*model[23:30])
		self.beblock3 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(inplace=True),
									  nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
		self.beblock2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
									  nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
		self.beblock1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
									  nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
		self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, bias=False), nn.ReLU(inplace=True))
		self.deblock = nn.Sequential(nn.Conv2d(896, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
									 nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
		
	def forward(self, x): # [1, 3, 1280, 1920]
		x_size = x.size()
		x = self.feblock(x) # [1, 64, 1280, 1920]
		x = self.feblock1(x) # [1, 256, 320, 480]
		x1 = x
		x = self.feblock2(x) # [1, 512, 160, 240]
		x2 = x
		x = self.feblock3(x) # [1, 512, 80, 120]
		x = self.beblock3(x) # [1, 512, 80, 120]
		x3_ = x
		x = upsample_bilinear(x, x2.shape) # [1, 512, 160, 240]
		x = torch.cat([x, x2], 1) # [1, 1024, 160, 240]
		x = self.beblock2(x) # [1, 256, 160, 240]
		x2_ = x
		x = upsample_bilinear(x, x1.shape) # [1, 256, 320, 480]
		x = torch.cat([x, x1], 1) # [1, 512, 320, 480]
		x1_ = self.beblock1(x) # [1, 128, 320, 480]
		x2_ = upsample_bilinear(x2_, x1.shape) # [1, 256, 320, 480]
		x3_ = upsample_bilinear(x3_, x1.shape) # [1, 512, 320, 480]
		x = torch.cat([x1_, x2_, x3_], 1) # [1, 896, 320, 480]
		x = self.deblock(x) # [1, 64, 320, 480]
		x = self.output_layer(x) # [1, 1, 320, 480]
		main_out = upsample_bilinear(x, size=x_size) # [1, 1, 1280, 1920]
		return main_out