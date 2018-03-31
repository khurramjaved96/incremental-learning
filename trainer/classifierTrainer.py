from __future__ import print_function

import copy

import torch
import torch.nn.functional as F
import torch.utils.data as td
from torch.autograd import Variable
import random

class trainer():
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, trainDataIteratorIdeal=None):
        self.trainDataIterator = trainDataIterator
        self.testDataIerator = testDataIterator
        self.trainDataIteratorIdeal = trainDataIteratorIdeal
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
        if args.ideal_nmc:
            self.trainLoaderIdeal = self.trainDataIteratorIdeal.dataset
        if not args.no_random:
            print("Randomly shuffling classes")
            random.seed(args.seed)
            random.shuffle(self.allClasses)

    def updateIterator(self, trainIterator):
        self.trainDataIterator = trainIterator
        self.trainLoader = trainIterator.dataset

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
            if self.args.ideal_nmc:
                self.trainDataIteratorIdeal.dataset.addClasses(popVal)
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
        if n not in self.olderClasses:
            self.olderClasses.append(n)

    def setupTraining(self):
        for param_group in self.optimizer.param_groups:
            print("Setting LR to", self.args.lr)
            param_group['lr'] = self.args.lr
            self.currentLr = self.args.lr

        k = 0
        if self.args.process == "nmc" and self.leftOver != []:
            k = int(self.args.memory_budget / len(self.leftOver))
        for val in self.leftOver:
            self.limitClass(val, k, not self.args.no_herding)


    def updateFrozenModel(self):
        self.model.eval()
        self.modelFixed = copy.deepcopy(self.model)
        for param in self.modelFixed.parameters():
            param.requires_grad = False

    def insert_generated_images(self, data, target, gan_images, gan_labels, batch_size, is_C=False):
        '''
        data: Images from data iterator
        target: Labels from data iterator
        gan_images: Generated images by GAN
        gan_labels: Python list containing unique classes of generated
                    images
        batch_size: Current batch_size of training iterator
        '''
        if self.args.process == 'gan' or self.args.process == 'cgan':
            if not len(gan_images) == 0:
                if is_C:
                    per_k_batch = (self.args.batch_size - batch_size) // len(gan_labels)
                    for k in gan_labels:
                        random_indices = torch.randperm(gan_images[k].shape[0])[0:per_k_batch]
                        new_targets = (torch.ones(per_k_batch) * k).long()
                        data = torch.cat((data, gan_images[k][random_indices]), dim=0)
                        target = torch.cat((target, new_targets), dim=0)
                else:
                    batch = self.args.batch_size - batch_size
                    gan_images = gan_images.cpu()
                    random_indices = torch.randperm(gan_images.shape[0])[0:batch]
                    data = torch.cat((data, gan_images[random_indices].data), dim=0)
                    target = None
        return data, target

    def train(self, gan_images=None, gan_labels=None, batch_size=None):
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

            y_onehot = torch.FloatTensor(len(data), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)

            output = self.model(Variable(data))
            if not self.args.no_distill:
                if len(self.olderClasses) > 0:
                    pred2 = self.modelFixed(Variable(data))
                    y_onehot[:, self.olderClasses] = pred2.data[:, self.olderClasses]

            loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss.backward()
            self.optimizer.step()

    #TODO Add generated images here
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