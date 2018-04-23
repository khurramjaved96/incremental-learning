import torch
import torch.nn as nn
import torch.nn.functional as F
from .gan_utils import normal_init

class Generator(nn.Module):
    def __init__(self, d=384, c=1, num_classes=10, nz=100):
        super(Generator, self).__init__()
        self.d = d
        self.nz = nz
        self.num_classes = num_classes
        self.fc1 = nn.Linear(nz+num_classes, d)

        self.ct2 = nn.ConvTranspose2d(d, d//2, 4, 1, 0, bias=False)
        self.ct2_bn = nn.BatchNorm2d(d//2)

        self.ct3 = nn.ConvTranspose2d(d//2, d//4, 4, 2, 1, bias=False)
        self.ct3_bn = nn.BatchNorm2d(d//4)

        self.ct4 = nn.ConvTranspose2d(d//4, d//8, 4, 2, 1, bias=False)
        self.ct4_bn = nn.BatchNorm2d(d//8)

        self.ct5 = nn.ConvTranspose2d(d//8, c, 4, 2, 1, bias=False)

        print(self)

    def forward(self, input):
        x = input.view(-1, self.nz + self.num_classes)
        x = self.fc1(x)
        x = x.view(-1, self.d, 1, 1)
        x = F.relu(self.ct2_bn(self.ct2(x)))
        x = F.relu(self.ct3_bn(self.ct3(x)))
        x = F.relu(self.ct4_bn(self.ct4(x)))
        x = F.tanh(self.ct5(x))
        return x

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std, False)

class Discriminator(nn.Module):
    def __init__(self, d=16, c=1, num_classes=10):
        super(Discriminator, self).__init__()
        self.d = d
        self.conv1 = nn.Conv2d(c, d, 3, 2, 1, bias=False)
        self.Drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(d, d*2, 3, 1, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.Drop2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(d*2, d*4, 3, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.Drop3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.Drop4 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(d*8, d*16, 3, 2, 1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(d*16)
        self.Drop5 = nn.Dropout(0.5)

        self.conv6 = nn.Conv2d(d*16, d*32, 3, 1, 1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(d*32)
        self.Drop6 = nn.Dropout(0.5)

        self.fc_dis = nn.Linear(4*4*d*32, 1)
        self.fc_aux = nn.Linear(4*4*d*32, num_classes)

        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        print(self)

    def forward(self, img, get_features=False):
        x = self.Drop1(F.leaky_relu(self.conv1(img), 0.2))
        x = self.Drop2(F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2))
        x = self.Drop3(F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2))
        x = self.Drop4(F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2))
        x = self.Drop5(F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2))
        x = self.Drop6(F.leaky_relu(self.conv6_bn(self.conv6(x)), 0.2))

        #When d=16, d*32=512, TODO
        x = x.view(-1, 4*4*self.d*32)
        fc_aux = self.fc_aux(x)
        if get_features:
            return fc_aux
        fc_dis = self.fc_dis(x)
        liklihood_correct_class = self.softmax(fc_aux)
        liklihood_real_img = self.sigmoid(fc_dis).view(-1,1).squeeze(1)
        return liklihood_real_img, liklihood_correct_class

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std, False)
