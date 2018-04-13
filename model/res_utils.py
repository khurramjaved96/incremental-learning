import torch
import torch.nn as nn

class DownsampleA(nn.Module):

  def __init__(self, n_in, n_out, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

class DownsampleC(nn.Module):

  def __init__(self, n_in, n_out, stride):
    super(DownsampleC, self).__init__()
    assert stride != 1 or n_in != n_out
    self.conv = nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, padding=0, bias=False)

  def forward(self, x):
    x = self.conv(x)
    return x

class DownsampleD(nn.Module):

  def __init__(self, n_in, n_out, stride):
    super(DownsampleD, self).__init__()
    assert stride == 2
    self.conv = nn.Conv2d(n_in, n_out, kernel_size=2, stride=stride, padding=0, bias=False)
    self.bn   = nn.BatchNorm2d(n_out)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x
