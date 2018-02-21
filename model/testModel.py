
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, noClasses, channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=5,padding=(2,2))
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5,padding=(2,2))
        self.conv2_bn1 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 10, kernel_size=5,padding=(2,2))
        self.conv2_bn2 = nn.BatchNorm2d(10)
        self.conv4 = nn.Conv2d(10, 14, kernel_size=5,padding=(2,2))
        self.conv2_bn3 = nn.BatchNorm2d(14)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(224, 100)
        self.fc2 = nn.Linear(100, noClasses)
        self.featureSize = 224
    def forward(self, x, feature=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_bn1(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv2_bn2(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_bn3(self.conv4(x))), 2))
        # print ("X after conv", x.shape)
        x = x.view(-1, 224)

        if feature:
            # print ("Size = ",torch.norm(x, 2, 1).unsqueeze(1).shape)
            return x / torch.norm(x, 2, 1).unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)

def testNetMNIST():
    return Net(10,1)