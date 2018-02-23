from __future__ import print_function

import copy
import random

import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GenericTrainer:
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        self.train_data_iterator = trainDataIterator
        self.test_data_iterator = testDataIterator
        self.model = model
        self.args = args
        self.dataset = dataset
        self.train_loader = self.train_data_iterator.dataset
        self.older_classes = []
        self.optimizer = optimizer
        self.model_fixed = copy.deepcopy(self.model)
        self.active_classes = []
        for param in self.model_fixed.parameters():
            param.requires_grad = False

        self.current_lr = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        self.left_over = []

        random.seed(args.seed)
        random.shuffle(self.all_classes)


class AutoEncoderTrainer(GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)

    def auto_encoder_model(self, noOfFeatures):
        '''
        :param noOfFeatures: No of features of the feature map. This is model dependant so not a constant
        :return: An auto-encoder that reduces the dimensions by a factor of 10. The auto encoder model has the same interface as 
        other models implemented in model module. 
        '''

        class AutoEncoderModelClass(nn.Module):
            def __init__(self, noOfFeatures):
                super(AutoEncoderModelClass, self).__init__()
                self.featureSize = int(noOfFeatures / 10)
                self.fc1 = nn.Linear(noOfFeatures, int(noOfFeatures / 10))
                self.fc2 = nn.Linear(int(noOfFeatures / 10), noOfFeatures)

            def forward(self, x, feature=False):
                x = F.sigmoid(self.fc1(x))
                if feature:
                    return x
                return self.fc2(x)

        myEncoder = AutoEncoderModelClass(noOfFeatures)
        if self.args.cuda:
            myEncoder.cuda()
        return myEncoder

    def train_auto_encoder(self, xIterator, epochs):
        bar = progressbar.ProgressBar()
        for epoch in range(epochs):
            for batch_idx, (data, target) in bar(enumerate(self.train_data_iterator)):
                pass

    def optimize(self, x, y, optimizer):
        pass


class Trainer(GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)


    def update_lr(self, epoch):
        if epoch == 10 :
            print ("Shifting from Adam to SGD")
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                      weight_decay=self.args.decay, nesterov=True)
        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from", self.current_lr, "to", self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self, classGroup):
        for temp in range(classGroup, classGroup + self.args.step_size):
            pop_val = self.all_classes.pop()
            self.train_data_iterator.dataset.add_class(pop_val)
            self.test_data_iterator.dataset.add_class(pop_val)
            print("Train Classes", self.train_data_iterator.dataset.active_classes)
            self.left_over.append(pop_val)

    def update_leftover(self, k):
        self.older_classes.append(k)

    def limit_class(self, n, k, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            print("Sorting by herding")
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)
        self.older_classes.append(n)

    def setup_training(self):
        print ("Shifting to Adam")
        self.optimizer = torch.optim.Adam(self.model.parameters())
        for param_group in self.optimizer.param_groups:
            print("Setting LR to", self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr

        for val in self.left_over:
            self.limit_class(val, int(self.args.memory_budget / len(self.left_over)), not self.args.no_herding)

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False


    def train(self):
        self.model.train()
        bar = progressbar.ProgressBar()
        for batch_idx, (data, target) in bar(enumerate(self.train_data_iterator)):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            weight_vector = (target * 0).int()
            for elem in self.older_classes:
                weight_vector = weight_vector + (target == elem).int()

            old_classes_indices = torch.squeeze(torch.nonzero((weight_vector > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((weight_vector == 0)).long())
            self.optimizer.zero_grad()

            if self.args.lwf:
                assert (len(old_classes_indices) == 0)
                assert (self.args.memory_budget == 0)

                y_onehot = torch.FloatTensor(len(target), self.dataset.classes)
                if self.args.cuda:
                    y_onehot = y_onehot.cuda()
                y_onehot.zero_()
                target.unsqueeze_(1)
                y_onehot.scatter_(1, target, 1)

                if len(self.older_classes) > 0:
                    pred2 = self.model_fixed(Variable(data))
                    data = torch.cat((data, data), dim=0)
                    output = self.model(Variable(data))
                    y_onehot = torch.cat((y_onehot, pred2.data), dim=0)

                else:
                    output = self.model(Variable(data))



            elif len(old_classes_indices) == 0:
                data_old_classes = data[new_classes_indices]
                targets_old_classes = target[new_classes_indices]
                target2 = targets_old_classes
                data_old_classes, target = Variable(data_old_classes), Variable(targets_old_classes)

                output = self.model(data_old_classes)
                y_onehot = torch.FloatTensor(len(data_old_classes), self.dataset.classes)
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
                    dataDis = Variable(data[old_classes_indices])
                    outpu2 = self.model_fixed(dataDis)
                    y_onehot[old_classes_indices] = outpu2.data

            loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss.backward()
            self.optimizer.step()
