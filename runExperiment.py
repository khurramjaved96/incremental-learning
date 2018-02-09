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

import torchvision

import dataHandler.incrementalLoaderCifar as dL
import model.modelFactory as mF
import copy
import plotter.plotter as plt
import trainer.classifierFactory as tF
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='argument to enable/disable distillation loss')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type',  default="resnet32",
                    help='model type to be used')
parser.add_argument('--name',  default="noname",
                    help='Name of the experiment')
parser.add_argument('--decay', type=float, default=0.00001 , help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--memory-budget', type=int, default=2000, help='How many images can we store at max')
parser.add_argument('--epochs-class', type=int, default=60, help='How many images can we store at max')
parser.add_argument('--classes', type=int, default=100, help='How many images can we store at max')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

schedule = args.schedule
gammas = args.gammas
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


experimentName = args.name+"_"+args.model_type+str(args.step_size)+str(args.memory_budget)
if args.no_distill:
    experimentName+= "_noDistill"

# Mean and STD of Cifar-100 dataset.
# To do : Remove the hard-coded mean and just compute it once using the data
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(0.1,0.1,0.1,0.1), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

train_data = datasets.CIFAR100("data", train=True, transform=train_transform, download=True)
test_data = datasets.CIFAR100("data", train=False, transform=test_transform, download=True)


trainDatasetFull = dL.incrementalLoaderCifar(train_data.train_data,train_data.train_labels, 500,100,[],transform=train_transform)
testDataset = dL.incrementalLoaderCifar(test_data.test_data,test_data.test_labels, 100,100,[],transform=test_transform)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader_full = torch.utils.data.DataLoader(trainDatasetFull,
    batch_size=args.batch_size, shuffle=True, **kwargs)


test_loader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


# Selecting model
myFactory = mF.modelFactory()
model = myFactory.getModel(args.model_type,args.classes)
if args.cuda:
    model.cuda()


modelFixed = None

y_onehot = torch.FloatTensor(args.batch_size, args.classes)
if args.cuda:
    y_onehot = y_onehot.cuda()


## Training code. To be moved to the trainer class
def train(epoch, optimizer, train_loader, leftover, verbose=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        weightVector = (target*0).int()
        for elem in leftover:
            weightVector = weightVector + (target==elem).int()

        weightVectorDis = torch.squeeze(torch.nonzero((weightVector>0)).long())
        print ("Length of vector", len(weightVectorDis))
        weightVectorNor = torch.squeeze(torch.nonzero((weightVector==0)).long())
        optimizer.zero_grad()
        targetTemp = target

        # Before incremental learing (without any distillation loss.)
        if len(weightVectorDis)==0:
            dataNorm = data[weightVectorNor]
            targetTemp = target
            targetNorm = target[weightVectorNor]
            target2 = targetNorm
            dataNorm, target = Variable(dataNorm), Variable(targetNorm)

            output = model(dataNorm)
            y_onehot = torch.FloatTensor(len(dataNorm), args.classes)
            if args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target2.unsqueeze_(1)
            y_onehot.scatter_(1, target2, 1)
            loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss.backward()
            optimizer.step()

        # After first increment. With distillation loss.
        elif args.no_distill:
            targetDis2 = targetTemp

            y_onehot = torch.FloatTensor(len(data), args.classes)
            if args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            targetDis2.unsqueeze_(1)
            y_onehot.scatter_(1, targetDis2, 1)

            output = model(Variable(data))
            # y_onehot[weightVectorDis] = outpu2.data
            #
            loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss.backward()
            optimizer.step()
        else:
          # optimizer.zero_grad()
            dataDis = Variable(data[weightVectorDis])
            targetDis2 = targetTemp

            y_onehot = torch.FloatTensor(len(data), args.classes)
            if args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            targetDis2.unsqueeze_(1)
            y_onehot.scatter_(1, targetDis2, 1)


            outpu2 = modelFixed(dataDis)
            output = model(Variable(data))
            y_onehot[weightVectorDis] = outpu2.data
#
            loss = F.binary_cross_entropy(output, Variable(y_onehot))

            loss.backward()
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

# Function to save confusion matrix. Can be helpful in visualizing what is going on.
def saveConfusionMatrix(epoch, path):
    model.eval()
    test_loss = 0
    correct = 0
    cMatrix = confusionmeter.ConfusionMeter(args.classes, True)

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if epoch > 0:
            cMatrix.add(pred, target.data.view_as(pred))

    test_loss /= len(test_loader.dataset)
    import cv2
    img = cMatrix.value() * 255
    cv2.imwrite(path + str(epoch) + ".jpg", img)
    return 100. * correct / len(test_loader.dataset)


def test(epoch, loader, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    if epoch >0:
        cMatrix = confusionmeter.ConfusionMeter(args.classes,True)

    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if epoch >0:
            cMatrix.add(pred, target.data.view_as(pred))


    test_loss /= len(loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return 100. * correct / len(loader.dataset)


optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                weight_decay=args.decay, nesterov=True)
currentLr = args.lr

# for epoch in range(1, args.epochs + 1):
allClasses = list(range(args.classes))
allClasses.sort(reverse=True)

# Will be important when computing confidence intervals.
import random
# random.shuffle(allClasses)

stepSize = args.step_size
leftOver = []
limitedset=[]
totalExmp = args.memory_budget
epochsPerClass=args.epochs_class
distillLoss = False


x = []
y = []
y1 = []

myTestFactory= tF.classifierFactory()
nmc = myTestFactory.getTester("nmc", args.cuda)

myPlotter = plt.plotter()
overallEpoch = 0
for classGroup in range(0, args.classes, stepSize):
    if classGroup ==0:
        distillLoss=False
    else:
        distillLoss=True
        modelFixed = copy.deepcopy(model)
        for param in modelFixed.parameters():
            param.requires_grad = False
        # model.classifier = nn.Linear(64, 100).cuda()
    for param_group in optimizer.param_groups:
        print ("Setting LR to", args.lr)
        param_group['lr'] = args.lr
        currentLr = args.lr 
    for val in leftOver:
        #print ("Limiting class", val,"to",int(totalExmp/len(leftOver)))
        trainDatasetFull.limitClass(val,int(totalExmp/len(leftOver)))
        limitedset.append(val)
    for temp in range(classGroup, classGroup+stepSize):
        popVal = allClasses.pop()
        trainDatasetFull.addClasses(popVal)
        testDataset.addClasses(popVal)
        leftOver.append(popVal)
    epoch=0
    for epoch in range(0,epochsPerClass):
        overallEpoch+=1
        for temp in range(0, len(schedule)):
            if schedule[temp]==epoch:
                for param_group in optimizer.param_groups:
                    currentLr = param_group['lr']
                    param_group['lr'] = currentLr*gammas[temp]
                    print("Changing learning rate from", currentLr, "to", currentLr*gammas[temp])
                    currentLr*= gammas[temp]
        train(int(classGroup/stepSize)*epochsPerClass + epoch,optimizer, train_loader_full,limitedset)
        if epoch%5==0:
            print("Train")
            test(int(classGroup / stepSize) * epochsPerClass + epoch, train_loader_full,  True)
            print ("Test")
            test(int(classGroup / stepSize) * epochsPerClass + epoch, test_loader, True)
    nmc.updateMeans(model, train_loader_full, args.cuda, args.classes)
    print ("Train Error", nmc.classify(model,train_loader_full,args.cuda, True))
    saveConfusionMatrix(int(classGroup/stepSize)*epochsPerClass + epoch,"../")
    print ("Test Error")
    y.append(nmc.classify(model,test_loader,args.cuda, True))
    y1.append(test(int(classGroup / stepSize) * epochsPerClass + epoch, test_loader, True))
    x.append(classGroup+stepSize)
    myPlotter.plot(x,y)
    myPlotter.plot(x, y1)

    myPlotter.saveFig("../"+experimentName+".jpg")
