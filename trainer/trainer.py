from __future__ import print_function

import copy
import random

import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
            # print("Train Classes", self.train_data_iterator.dataset.active_classes)
            self.left_over.append(pop_val)

    def update_leftover(self, k):
        self.older_classes.append(k)

    def limit_class(self, n, k, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            # print("Sorting by herding")
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)
        self.older_classes.append(n)

    def setup_training(self):
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
        self.model_fixed.eval()


    def train(self, epoch):

        self.model.train()
        length_epoch = self.dataset.labels_per_class_train*(len(self.train_data_iterator.dataset.active_classes)-len(self.train_data_iterator.dataset.limited_classes))
        length_epoch += self.args.memory_budget
        length_epoch/= int(self.args.batch_size)
        print("Current Epoch : ", epoch, "size : ", length_epoch)
        bar = progressbar.ProgressBar(redirect_stdout=True)
        counter=0
        for batch_idx, (data, target) in enumerate(self.train_data_iterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()


            counter+=1

            weight_vector = (target * 0).int()
            for elem in self.older_classes:
                weight_vector = weight_vector + (target == elem).int()

            old_classes_indices = torch.squeeze(torch.nonzero((weight_vector > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((weight_vector == 0)).long())

            # Compute weight for each instance; weight is proportional to no of samples of the class in the training set.

            daTemp = np.ones((self.args.batch_size))
            weight = torch.FloatTensor(daTemp).cpu()

            # print ("Weight", weight)

            OrigSize = self.dataset.labels_per_class_train
            if len(self.older_classes)>0:
                ChangedSize = min(self.args.memory_budget//len(self.older_classes), OrigSize)
                weight[old_classes_indices] = 1.0 / float(ChangedSize)

            weight[new_classes_indices] = 1.0/float(OrigSize)

            if epoch==0 and batch_idx==0:
                print (weight.cpu().numpy())

            self.optimizer.zero_grad()

            y_onehot = torch.FloatTensor(len(target), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)

            output = self.model(Variable(data))
            if len(self.older_classes) > 0:
                pred2 = self.model_fixed(Variable(data))
                y_onehot[:, self.older_classes] = pred2.data[:, self.older_classes]

            # if self.args.lwf:
            #
            #     assert (len(old_classes_indices) == 0)
            #     assert (self.args.memory_budget == 0)
            #
            #     if len(self.older_classes) > 0:
            #         if epoch == 0 and batch_idx == 0 :
            #             print("Warm up step for 2 epochs")
            #             for param in self.model.named_parameters():
            #                 if "fc" in param[0]:
            #                     print ("Setting gradient of FC layers to be True")
            #                     param[1].requies_grad = True
            #                 else:
            #                     param[1].requires_grad = False
            #         if epoch == 2 and batch_idx == 0:
            #             print("Shifting to all weight training from warm up training")
            #             for param in self.model.parameters():
            #                 param.requires_grad = True
            #
            #         pred2 = self.model_fixed(Variable(data))
            #         output = self.model(Variable(data))
            #         y_onehot[:, self.older_classes] = pred2.data[:, self.older_classes]
            #
            #     else:
            #         output = self.model(Variable(data))
            #
            #
            # else:
            # print (data.shape)
            # print (y_onehot.shape)

            if self.args.cuda:
                weight.cuda()

            if self.args.no_upsampling:
                loss = F.binary_cross_entropy(output, Variable(y_onehot),weight.unsqueeze(1))
            else:
                loss = F.binary_cross_entropy(output, Variable(y_onehot), weight.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            bar.update(int(float(batch_idx+1)/float(len(self.train_data_iterator))*100))
