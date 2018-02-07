
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, noClasses):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5,padding=(2,2))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5,padding=(2,2))
        self.conv2_bn1 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5,padding=(2,2))
        self.conv2_bn2 = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=5,padding=(2,2))
        self.conv2_bn3 = nn.BatchNorm2d(40)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(640, 100)
        self.fc2 = nn.Linear(100, noClasses)

    def forward(self, x, feature=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_bn1(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv2_bn2(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_bn3(self.conv4(x))), 2))
        # print ("X after conv", x.shape)
        x = x.view(-1, 640)
        if feature:
            return x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)