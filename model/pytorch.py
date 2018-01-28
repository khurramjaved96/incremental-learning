from __future__ import print_function
import argparse
import torch.utils.data as td
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchnet.meter import confusionmeter

import numpy as np
import utils

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

train_data = datasets.CIFAR100("../data", train=True, transform=train_transform, download=True)
test_data = datasets.CIFAR100("../data", train=False, transform=test_transform, download=True)


trainDataset = utils.incrementalLoaderCifar(train_data.train_data,train_data.train_labels, 500,100,list(range(10)),transform=train_transform)
testDataset = utils.incrementalLoaderCifar(test_data.test_data,test_data.test_labels, 100,100,list(range(10)),transform=test_transform)



kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(trainDataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)



class Net(nn.Module):
    def __init__(self):
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
        self.fc2 = nn.Linear(100, 100 )

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_bn1(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv2_bn2(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_bn3(self.conv4(x))), 2))
        # print ("X after conv", x.shape)
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # print ("X Shape", x.shape)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())



def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch=0):
    model.eval()
    test_loss = 0
    correct = 0
    if epoch >0:
        cMatrix = confusionmeter.ConfusionMeter(100,True)

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if epoch >0:
            cMatrix.add(pred, target.data.view_as(pred))


    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if epoch >0:
        import cv2
        img = utils.resizeImage(cMatrix.value(), 10)*255
        cv2.imwrite("/output/Image"+str(epoch)+".jpg", img)




for epoch in range(1, args.epochs + 1):

    if epoch==100:
        for a in range(10,20):
            trainDataset.addClasses(a)
            testDataset.addClasses(a)
        for a in range(0,10):
            trainDataset.limitClass(a,30)
    if epoch == 150:
        for a in range(20, 30):
            trainDataset.addClasses(a)
            testDataset.addClasses(a)
        for a in range(10, 20):
            trainDataset.limitClass(a, 30)
    train(epoch)
    if epoch%10==0:
        test(epoch)
    else:
        test(epoch)

