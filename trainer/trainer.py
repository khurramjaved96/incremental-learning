from __future__ import print_function

import copy

import torch
import torch.nn.functional as F
import torch.utils.data as td
from torch.autograd import Variable
import random

class trainer():
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        self.trainDataIterator = trainDataIterator
        self.testDataIerator = testDataIterator
        self.model = model
        self.args = args
        self.dataset = dataset
        self.trainLoader = self.trainDataIterator.dataset
        self.olderClasses = []
        self.optimizer = optimizer
        self.modelFixed = copy.deepcopy(self.model)
        self.activeClasses = []
        for param in self.modelFixed.parameters():
            param.requires_grad = False

        self.currentLr = args.lr
        self.allClasses = list(range(dataset.classes))
        self.allClasses.sort(reverse=True)
        self.leftOver = []
        if not args.no_random:
            print("Randomly shuffling classes")
            random.shuffle(self.allClasses)

    def updateLR(self, epoch):
        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.currentLr = param_group['lr']
                    param_group['lr'] = self.currentLr * self.args.gammas[temp]
                    print("Changing learning rate from", self.currentLr, "to", self.currentLr * self.args.gammas[temp])
                    self.currentLr *= self.args.gammas[temp]

    def incrementClasses(self, classGroup):
        for temp in range(classGroup, classGroup + self.args.step_size):
            popVal = self.allClasses.pop()
            self.trainDataIterator.dataset.addClasses(popVal)
            self.testDataIerator.dataset.addClasses(popVal)
            print("Train Classes", self.trainDataIterator.dataset.activeClasses)
            self.leftOver.append(popVal)

    def updateLeftover(self, k):
        self.olderClasses.append(k)

    def limitClass(self, n, k, herding=True):
        if not herding:
            self.trainLoader.limitClass(n, k)
        else:
            print("Sorting by herding")
            self.trainLoader.limitClassAndSort(n, k, self.modelFixed)
        self.olderClasses.append(n)

    def setupTraining(self):
        for param_group in self.optimizer.param_groups:
            print("Setting LR to", self.args.lr)
            param_group['lr'] = self.args.lr
            self.currentLr = self.args.lr

        for val in self.leftOver:
            self.limitClass(val, int(self.args.memory_budget / len(self.leftOver)), not self.args.no_herding)


    def updateFrozenModel(self):
        self.modelFixed = copy.deepcopy(self.model)
        for param in self.modelFixed.parameters():
            param.requires_grad = False

    def train(self):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.trainDataIterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            weightVector = (target * 0).int()
            for elem in self.olderClasses:
                weightVector = weightVector + (target == elem).int()

            oldClassesIndices = torch.squeeze(torch.nonzero((weightVector > 0)).long())
            newClassesIndices = torch.squeeze(torch.nonzero((weightVector == 0)).long())
            self.optimizer.zero_grad()

            if len(oldClassesIndices) == 0:
                dataOldClasses = data[newClassesIndices]
                targetsOldClasses = target[newClassesIndices]
                target2 = targetsOldClasses
                dataOldClasses, target = Variable(dataOldClasses), Variable(targetsOldClasses)

                output = self.model(dataOldClasses)
                y_onehot = torch.FloatTensor(len(dataOldClasses), self.dataset.classes)
                if self.args.cuda:
                    y_onehot = y_onehot.cuda()

                y_onehot.zero_()
                target2.unsqueeze_(1)
                y_onehot.scatter_(1, target2, 1)

            else:
                y_onehot = torch.FloatTensor(len(data), self.dataset.classes)
                if self.args.cuda:
                    y_onehot = y_onehot.cuda()

                y_onehot.zero_()
                target.unsqueeze_(1)
                y_onehot.scatter_(1, target, 1)

                output = self.model(Variable(data))
                if not self.args.no_distill:
                    dataDis = Variable(data[oldClassesIndices])
                    outpu2 = self.modelFixed(dataDis)
                    y_onehot[oldClassesIndices] = outpu2.data

            loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss.backward()
            self.optimizer.step()

    def evaluate(self, loader):
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, target in loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(loader.dataset)
        return 100. * correct / len(loader.dataset)


def train(optimizer, train_loader, leftover, model, modelFixed, args, dataset, verbose=False):
    model.train()

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if not args.oversampling:
        # print (train_loader.dataset.weights)
        train_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                   sampler=torch.utils.data.sampler.WeightedRandomSampler(
                                                       train_loader.dataset.weights.tolist(),
                                                       len(
                                                           train_loader.dataset.activeClasses) * train_loader.dataset.classSize),
                                                   batch_size=args.batch_size,
                                                   **kwargs)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        weightVector = (target * 0).int()
        for elem in leftover:
            weightVector = weightVector + (target == elem).int()

        oldClassesIndices = torch.squeeze(torch.nonzero((weightVector > 0)).long())
        newClassesIndices = torch.squeeze(torch.nonzero((weightVector == 0)).long())
        optimizer.zero_grad()

        if len(oldClassesIndices) == 0:
            dataOldClasses = data[newClassesIndices]
            targetsOldClasses = target[newClassesIndices]
            target2 = targetsOldClasses
            dataOldClasses, target = Variable(dataOldClasses), Variable(targetsOldClasses)

            output = model(dataOldClasses)
            y_onehot = torch.FloatTensor(len(dataOldClasses), dataset.classes)
            if args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target2.unsqueeze_(1)
            y_onehot.scatter_(1, target2, 1)

        else:
            y_onehot = torch.FloatTensor(len(data), dataset.classes)
            if args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)

            output = model(Variable(data))
            if not args.no_distill:
                dataDis = Variable(data[oldClassesIndices])
                outpu2 = modelFixed(dataDis)
                y_onehot[oldClassesIndices] = outpu2.data

        loss = F.binary_cross_entropy(output, Variable(y_onehot))
        loss.backward()
        optimizer.step()


def test(loader, model, args):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    return 100. * correct / len(loader.dataset)
