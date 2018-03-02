import torch
import torch.nn as nn
import torch.nn.functional as F
from .gan_utils import normal_init

class Generator(nn.Module):
    def __init__(self, d=128, c=1):
        super(Generator, self).__init__()
        #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.ct1_noise = nn.ConvTranspose2d(100, d*4, 4, 1, 0)
        self.ct1_noise_bn = nn.BatchNorm2d(d*4)
        self.ct2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.ct2_bn = nn.BatchNorm2d(d*2)
        self.ct3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.ct3_bn = nn.BatchNorm2d(d)
        self.ct4 = nn.ConvTranspose2d(d, c, 4, 2, 1)

    def forward(self, noise):
        x = F.relu(self.ct1_noise_bn(self.ct1_noise(noise)))
        x = F.relu(self.ct2_bn(self.ct2(x)))
        x = F.relu(self.ct3_bn(self.ct3(x)))
        x = F.tanh(self.ct4(x))
        return x

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class Discriminator(nn.Module):
    def __init__(self, d=128, c=1):
        super(Discriminator, self).__init__()
        self.conv1_img = nn.Conv2d(c, d//2, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d//2)
        self.conv2 = nn.Conv2d(d//2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def forward(self, img):
        x = F.leaky_relu(self.conv1_bn(self.conv1_img(img)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
