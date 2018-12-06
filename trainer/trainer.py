''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

import model

logger = logging.getLogger('iCARL')


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

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
        self.models = []
        self.current_lr = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        self.left_over = []
        self.ideal_iterator = ideal_iterator
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None

        logger.warning("Shuffling turned off for debugging")
        # random.seed(args.seed)
        # random.shuffle(self.all_classes)


class Trainer(GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator)
        self.dynamic_threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.gradient_threshold_unreported_experiment = np.ones(self.dataset.classes, dtype=np.float64)

    def update_lr(self, epoch):
        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    logger.debug("Changing learning rate from %0.2f to %0.2f", self.current_lr,
                                 self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self, class_group):
        '''
        Add classes starting from class_group to class_group + step_size 
        :param class_group: 
        :return: N/A. Only has side-affects 
        '''
        for temp in range(class_group, class_group + self.args.step_size):
            pop_val = self.all_classes.pop()
            self.train_data_iterator.dataset.add_class(pop_val)
            self.ideal_iterator.dataset.add_class(pop_val)
            self.test_data_iterator.dataset.add_class(pop_val)
            self.left_over.append(pop_val)

    def limit_class(self, n, k, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)
        if n not in self.older_classes:
            self.older_classes.append(n)

    def reset_dynamic_threshold(self):
        '''
        Reset the threshold vector maintaining the scale factor. 
        Important to set this to zero before every increment. 
        setupTraining() also does this so not necessary to call both. 
        :return: 
        '''
        threshTemp = self.dynamic_threshold / np.max(self.dynamic_threshold)
        threshTemp = ['{0:.4f}'.format(i) for i in threshTemp]

        threshTemp2 = self.gradient_threshold_unreported_experiment / np.max(
            self.gradient_threshold_unreported_experiment)
        threshTemp2 = ['{0:.4f}'.format(i) for i in threshTemp2]

        logger.debug("Scale Factor" + ",".join(threshTemp))
        logger.debug("Scale GFactor" + ",".join(threshTemp2))

        self.dynamic_threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.gradient_threshold_unreported_experiment = np.ones(self.dataset.classes, dtype=np.float64)

    def setup_training(self):
        self.reset_dynamic_threshold()

        for param_group in self.optimizer.param_groups:
            logger.debug("Setting LR to %0.2f", self.args.lr)
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
            logger.warning("Random Initilization of weights at each increment")
            myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
            if self.args.cuda:
                myModel.cuda()
            self.model = myModel
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.decay, nesterov=True)
            self.model.eval()

    def randomly_init_model(self):
        logger.info("Randomly initilizaing model")
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
        if self.args.cuda:
            myModel.cuda()
        self.model = myModel
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=True)
        self.model.eval()

    def get_model(self):
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
        logger.info("Epochs %d", epoch)
        for data, y, target in tqdm(self.train_data_iterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
                y = y.cuda()
            oldClassesIndices = (target * 0).int()
            for elem in range(0, self.args.unstructured_size):
                oldClassesIndices = oldClassesIndices + (target == elem).int()

            old_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices == 0)).long())

            self.optimizer.zero_grad()

            target_normal_loss = target[new_classes_indices]
            data_normal_loss = data[new_classes_indices]

            target_distillation_loss = y.float()
            data_distillation_loss = data

            y_onehot = torch.FloatTensor(len(target_normal_loss), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target_normal_loss.unsqueeze_(1)
            y_onehot.scatter_(1, target_normal_loss, 1)

            output = self.model(Variable(data_normal_loss))
            self.dynamic_threshold += np.sum(y_onehot.cpu().numpy(), 0)
            loss = F.kl_div(output, Variable(y_onehot))

            myT = self.args.T

            if self.args.no_distill:
                pass

            elif len(self.older_classes) > 0:
                # Get softened labels of the model from a previous version of the model.
                pred2 = self.model_fixed(Variable(data_distillation_loss), T=myT, labels=True).data
                # Softened output of the model
                if myT > 1:
                    output2 = self.model(Variable(data_distillation_loss), T=myT)
                else:
                    output2 = output

                self.dynamic_threshold += (np.sum(pred2.cpu().numpy(), 0)) * (
                        myT * myT) * self.args.alpha
                loss2 = F.kl_div(output2, Variable(pred2))

                loss2.backward(retain_graph=True)

                # Scale gradient by a factor of square of T. See Distilling Knowledge in Neural Networks by Hinton et.al. for details.
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad * (myT * myT) * self.args.alpha

            if len(self.older_classes) == 0 or not self.args.no_nl:
                loss.backward()

            for param in self.model.named_parameters():
                if "fc.weight" in param[0]:
                    self.gradient_threshold_unreported_experiment *= 0.99
                    self.gradient_threshold_unreported_experiment += np.sum(np.abs(param[1].grad.data.cpu().numpy()), 1)

            self.optimizer.step()

        if self.args.no_nl:
            self.dynamic_threshold[len(self.older_classes):len(self.dynamic_threshold)] = np.max(self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[
            len(self.older_classes):len(self.gradient_threshold_unreported_experiment)] = np.max(
                self.gradient_threshold_unreported_experiment)
        else:
            self.dynamic_threshold[0:self.args.unstructured_size] = np.max(self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[0:self.args.unstructured_size] = np.max(
                self.gradient_threshold_unreported_experiment)

            self.dynamic_threshold[self.args.unstructured_size + len(
                self.older_classes) + self.args.step_size: len(self.dynamic_threshold)] = np.max(
                self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[self.args.unstructured_size + len(
                self.older_classes) + self.args.step_size: len(self.gradient_threshold_unreported_experiment)] = np.max(
                self.gradient_threshold_unreported_experiment)

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        logger.debug("Total Models %d", len(self.models))
