import torch
import torch.nn as nn
import torch.nn.functional as F
from .gan_utils import normal_init

# "Directly applying batchnorm to all layers however,
# resulted in sample oscillation and model instability.
# This was avoided by not applying batchnorm to the
# generator output layer and the discriminator input layer."
# DCGAN paper https://arxiv.org/pdf/1511.06434.pdf

#TODO Add the jump incase of d=128 as in original WGAN
class Generator(nn.Module):
    def __init__(self, d=64, c=1):
        super(Generator, self).__init__()
        #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.ct1_noise = nn.ConvTranspose2d(100, d*4, 4, 1, 0)
        self.ct1_noise_bn = nn.BatchNorm2d(d*4)
        self.ct2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.ct2_bn = nn.BatchNorm2d(d*2)
        self.ct3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.ct3_bn = nn.BatchNorm2d(d)
        self.ct4 = nn.ConvTranspose2d(d, c, 4, 2, 1)
        print(self)

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
    def __init__(self, d=64, c=1):
        super(Discriminator, self).__init__()
        self.conv1_img = nn.Conv2d(c, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)
        #self.conv4_bn = nn.BatchNorm2d(d*8)
        #self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        print(self)

    def forward(self, img):
        #Note: no sigmoid for wgan
        x = F.leaky_relu(self.conv1_img(img), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        #x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv4(x)
        x = x.mean(0)
        return x.view(1)

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
