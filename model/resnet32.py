import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .res_utils import DownsampleA


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.featureSize = 64

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes, channels=3):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()

        self.featureSize = 64
        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False, T=1, labels=False, getAllFeatures = False, scale=None):
        # print ("X shape", x.shape)
        if getAllFeatures:
            x1 = self.conv_1_3x3(x)
            x2 = F.relu(self.bn_1(x1), inplace=True)
            x3 = self.stage_1(x2)
            x3Features = x3.view(x3.size(0), -1)
            # print("X3 shape", x3Features.shape)
            x4 = self.stage_2(x3)
            x4Features = x4.view(x4.size(0), -1)
            # print("X4 shape", x4Features.shape)
            x5 = self.stage_3(x4)
            x5Features = x5.view(x5.size(0), -1)
            # print("X5 shape", x5Features.shape)
            x6 = self.avgpool(x5)
            x7 = x6.view(x6.size(0), -1)
            # print ("X7 shape", x7.shape)
            finalFeature = torch.cat((x3Features, x4Features, x5Features, x7), dim=1)
            # print (finalFeature.shape)
            # 0/0
            return finalFeature


        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if feature:
            return x / torch.norm(x, 2, 1).unsqueeze(1)
        if labels:
            return F.softmax(self.fc(x)/T)
        if scale is not None:
            x = self.fc(x)
            x = x
            temp = F.log_softmax(x / T)
            # print("Gets here; scaled output")
            return temp*scale
            return temp / scale
        return F.log_softmax(self.fc(x)/T)

    def forwardFeature(self, x):
        pass


def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model


def resnet10mnist(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 10, num_classes, 1)
    return model


def resnet20mnist(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes, 1)
    return model


def resnet32mnist(num_classes=10, channels=1):
    model = CifarResNet(ResNetBasicblock, 32, num_classes, channels)
    return model


def resnet32(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    return model


def resnet44(num_classes=10):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 44, num_classes)
    return model


def resnet56(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    return model


def resnet110(num_classes=10):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 110, num_classes)
    return model
