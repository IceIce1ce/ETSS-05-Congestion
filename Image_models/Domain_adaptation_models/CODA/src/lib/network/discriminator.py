import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(Discriminator, self).__init__()
		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self._init_weights()

	def _init_weights(self, truncate=False):
		def normal_init(m, mean, stddev, truncated=False):
			if truncated:
				m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
			else:
				m.weight.data.normal_(mean, stddev)
				m.bias.data.zero_()
		normal_init(self.conv1, 0, 0.01, truncate)
		normal_init(self.conv2, 0, 0.01, truncate)
		normal_init(self.conv3, 0, 0.01, truncate)
		normal_init(self.conv4, 0, 0.01, truncate)
		normal_init(self.classifier, 0, 0.01, truncate)

	def forward(self, x): # [16, 1, 512, 512]
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x) # [16, 1, 16, 16]
		return x