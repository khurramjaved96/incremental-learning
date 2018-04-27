from __future__ import print_function

import copy
import random

import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import utils.utils as ut
import torchvision
import model

class GenericTrainer:
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None):
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
        self.models= []
        self.current_lr = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        self.left_over = []
        self.ideal_iterator = ideal_iterator
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None

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
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator)
        self.threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.threshold2 = np.ones(self.dataset.classes, dtype=np.float64)

    def update_lr(self, epoch):
        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from", self.current_lr, "to",
                          self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self, classGroup):
        for temp in range(classGroup, classGroup + self.args.step_size):
            pop_val = self.all_classes.pop()
            self.train_data_iterator.dataset.add_class(pop_val)
            self.ideal_iterator.dataset.add_class(pop_val)
            self.test_data_iterator.dataset.add_class(pop_val)
            # print("Train Classes", self.train_data_iterator.dataset.active_classes)
            self.left_over.append(pop_val)

    def increment_classes_2(self, start, end):
        for temp in range(start, end):
            pop_val = self.all_classes.pop()
            self.train_data_iterator.dataset.add_class(pop_val)

            self.ideal_iterator.dataset.add_class(pop_val)
            self.ideal_iterator.dataset.limit_class(pop_val, 0)

            self.test_data_iterator.dataset.add_class(pop_val)
            self.test_data_iterator.dataset.limit_class(pop_val, 0)
            # print("Train Classes", self.train_data_iterator.dataset.active_classes)




    def limit_class(self, n, k, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            # print("Sorting by herding")
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)
        if n not in self.older_classes:
            self.older_classes.append(n)

    def setup_training(self):
        print("Threshold", self.threshold / np.max(self.threshold))
        print("Threshold 2", self.threshold2 / np.max(self.threshold2))
        self.threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.threshold2 = np.ones(self.dataset.classes, dtype=np.float64)

        # self.args.alpha += self.args.alpha_increment
        for param_group in self.optimizer.param_groups:
            print("Setting LR to", self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr
        for val in self.left_over:
            self.limit_class(val, int(self.args.memory_budget / len(self.left_over)), not self.args.no_herding)


    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.models.append(self.model_fixed)

        if self.args.random_init:
            print ("Random Initilization of weights")
            myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
            if self.args.cuda:
                myModel.cuda()
            self.model = myModel
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.decay, nesterov=True)
            self.model.eval()

    def getModel(self):
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
        if self.args.cuda:
            myModel.cuda()
        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer


    def train(self, epoch):

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_data_iterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            oldClassesIndices = (target * 0).int()
            for elem in range(0, self.args.unstructured_size):
                oldClassesIndices = oldClassesIndices + (target == elem).int()

            old_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices == 0)).long())

            self.optimizer.zero_grad()

            target_normal_loss = target[new_classes_indices]
            data_normal_loss = data[new_classes_indices]

            target_distillation_loss = target[old_classes_indices]
            data_distillation_loss = data[old_classes_indices]

            y_onehot = torch.FloatTensor(len(target_normal_loss), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target_normal_loss.unsqueeze_(1)
            y_onehot.scatter_(1, target_normal_loss, 1)

            if len(self.older_classes) == 0 or not self.args.no_nl:
                output = self.model(Variable(data_normal_loss))
                self.threshold += np.sum(y_onehot.cpu().numpy(), 0) / len(target_normal_loss.cpu().numpy())
                loss = F.kl_div(output, Variable(y_onehot))
            myT = self.args.T

            if self.args.no_distill:
                pass

            elif len(self.older_classes) > 0:

                # Get softened targets generated from previous mode2l;a
                tempModel = np.random.choice(self.models)

                pred2 = tempModel(Variable(data_distillation_loss), T=myT, labels=True)
                # Softened output of the model
                output2 = self.model(Variable(data_distillation_loss), T=myT)

                # output2_t, output3_t = self.model(Variable(data3), T=myT, labels=True, logits=True)


                self.threshold += (np.sum(pred2.data.cpu().numpy(), 0) / len(data_distillation_loss.cpu().numpy())) * (
                myT * myT) * self.args.alpha*len(self.models)
                loss2 = F.kl_div(output2, Variable(pred2.data))

                loss2.backward(retain_graph=True)

                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad * (myT * myT) * self.args.alpha*len(self.models)

            if len(self.older_classes) == 0 or not self.args.no_nl:
                loss.backward()

            for param in self.model.named_parameters():
                if "fc.weight" in param[0]:
                    self.threshold2 *= 0.99
                    self.threshold2 += np.sum(np.abs(param[1].grad.data.cpu().numpy()), 1)

            self.optimizer.step()

        if self.args.no_nl:
            self.threshold[len(self.older_classes):len(self.threshold)] = np.max(self.threshold)
            self.threshold2[len(self.older_classes):len(self.threshold2)] = np.max(self.threshold2)
        else:
            self.threshold[0:self.args.unstructured_size] = np.max(self.threshold)
            self.threshold2[0:self.args.unstructured_size] = np.max(self.threshold2)

            self.threshold[self.args.unstructured_size + len(
                self.older_classes) + self.args.step_size:self.args.unstructured_size + len(self.threshold)] = np.max(
                self.threshold)
            self.threshold2[self.args.unstructured_size + len(
                self.older_classes) + self.args.step_size:self.args.unstructured_size + len(self.threshold2)] = np.max(
                self.threshold2)


    def addModel(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        print ("Total Models", len(self.models))

    def trainSingle(self, epoch):

        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer_single.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from", self.current_lr, "to",
                          self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

        self.model_single.train()

        for batch_idx, (data, target) in enumerate(self.train_data_iterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()


            oldClassesIndices = (target * 0).int()
            for elem in range(0,self.args.unstructured_size):
                oldClassesIndices = oldClassesIndices + (target == elem).int()

            new_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices == 0)).long())

            if len(new_classes_indices)>0:
                self.optimizer_single.zero_grad()

                target_normal_loss = target[new_classes_indices]
                data_normal_loss = data[new_classes_indices]

                y_onehot = torch.FloatTensor(len(target_normal_loss), self.dataset.classes)
                if self.args.cuda:
                    y_onehot = y_onehot.cuda()

                y_onehot.zero_()
                target_normal_loss.unsqueeze_(1)
                y_onehot.scatter_(1, target_normal_loss, 1)

                output = self.model_single(Variable(data_normal_loss))
                loss = F.kl_div(output, Variable(y_onehot))

                loss.backward()

                self.optimizer_single.step()


