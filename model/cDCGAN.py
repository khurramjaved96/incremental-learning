import torch
import torch.nn as nn
import torch.nn.functional as F
from .gan_utils import normal_init

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*shape)

class Generator(nn.Module):
    '''
    d = base multiplier
    c = number of channels in the image
    l = number of unique classes in the dataset
    '''
    def __init__(self, d=128, c=1, l=10):
        super(Generator, self).__init__()
        #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.ct1_noise = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.ct1_noise_bn = nn.BatchNorm2d(d*2)
        self.ct1_label = nn.ConvTranspose2d(l, d*2, 4, 1, 0)
        self.ct1_label_bn = nn.BatchNorm2d(d*2)
        self.ct2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.ct2_bn = nn.BatchNorm2d(d*2)
        self.ct3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.ct3_bn = nn.BatchNorm2d(d)
        self.ct4 = nn.ConvTranspose2d(d, c, 4, 2, 1)

    def forward(self, noise, label):
        x = F.relu(self.ct1_noise_bn(self.ct1_noise(noise)))
        y = F.relu(self.ct1_label_bn(self.ct1_label(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.ct2_bn(self.ct2(x)))
        x = F.relu(self.ct3_bn(self.ct3(x)))
        x = F.tanh(self.ct4(x))
        return x

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):
    '''
    d = base multiplier
    c = number of channels in the image
    l = number of unique classes in the dataset
    '''
    def __init__(self, d=128, c=1, l=10, use_mbd=True, mbd_num=128, mbd_dim=3):
        super(Discriminator, self).__init__()
        self.use_mbd = use_mbd
        self.mbd_num = mbd_num
        self.mbd_dim = mbd_dim

        self.conv1_img = nn.Conv2d(c, d//2, 4, 2, 1)
        self.conv1_label = nn.Conv2d(l, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        # Use linear layer to produce a matrix of shape (B, mbd_num*mbd_dim)
        # The input to this layer is of shape [B, d*4, 4, 4], we reshape it
        if self.use_mbd:
            self.mbd = nn.Linear(d*4*4*4, mbd_num * mbd_dim)
        self.conv4 = nn.Conv2d(d * 4 + 8, 1, 4, 1, 0)

    def forward(self, img, label):
        x = F.leaky_relu(self.conv1_img(img), 0.2)
        y = F.leaky_relu(self.conv1_label(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        if self.use_mbd:
            # Reshape for linear layer
            x = x.view(-1, 128 * 4 * 4 * 4)
            # Use the linear layer
            mbd = self.mbd(x)
            # Calculate minibatch features and concat them
            x = self.minibatch_discrimination(mbd, x)
            # Make it compatible for convolution layer
            # 520 = 128 * 4 + 8
            x = x.view(-1, 520, 4, 4)
        x = self.conv4(x)
        x = F.sigmoid(x)
        return x

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def minibatch_discrimination(self, x, input_to_layer):
        activation = x.view(-1, self.mbd_num, self.mbd_dim)
        diffs = activation.unsqueeze(3) - activation.permute(1,2,0).unsqueeze(0)
        abs_diff = torch.abs(diffs).sum(2)
        mb_feats = torch.exp(-abs_diff).sum(2)
        return torch.cat([input_to_layer, mb_feats], 1)
