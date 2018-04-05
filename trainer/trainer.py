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

        self.current_lr = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        self.left_over = []
        self.ideal_iterator = ideal_iterator

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

    def convert_to_adversarial_instance(self, instance, target_class, required_confidence = 0.90, alpha = 1, iters = 100):
        instance.unsqueeze_(0)
        instance = Variable(instance, requires_grad=True)

        ce_loss = nn.CrossEntropyLoss()
        im_label_as_var = Variable(torch.from_numpy(np.asarray([target_class])))

        #self.model_fixed.eval()
        for i in range(1, iters):
            instance.grad = None

            # Forward
            output = self.model_fixed(instance)

            # Get confidence
            target_confidence = (output)[0][target_class].data.numpy()[0]
            print('Iteration:', str(i), 'Target Confidence', "{0:.4f}".format(target_confidence))
            if target_confidence > required_confidence:
                break

            # Zero grads
            self.model_fixed.zero_grad()

            # Backward
            pred_loss = ce_loss(output, im_label_as_var)
            pred_loss.backward()

            # Update instance with adversarial noise
            adv_noise = alpha * torch.sign(instance.grad.data)
            instance.data = instance.data - adv_noise
        #self.model_fixed.train()

        return torch.from_numpy(instance.data.cpu().numpy().squeeze(0)).float()

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

        for batch_idx, (data, target) in enumerate(self.train_data_iterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            weight_vector = (target * 0).int()
            for elem in self.older_classes:
                weight_vector = weight_vector + (target == elem).int()

            # Use this to implement decayed distillation

            old_classes_indices = torch.squeeze(torch.nonzero((weight_vector > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((weight_vector == 0)).long())

            self.optimizer.zero_grad()

            if self.args.rand or self.args.adversarial:
                for old in old_classes_indices:
                    if self.args.rand:
                        data[old] = self.dataset.get_random_instance()
                    elif self.args.adversarial:
                        data[old] = self.convert_to_adversarial_instance(self.dataset.get_random_instance(), target[old])

            y_onehot = torch.FloatTensor(len(target), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)

            output = self.model(Variable(data))
            # loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss = F.kl_div(output, Variable(y_onehot))

            myT = self.args.T
            if self.args.no_distill:
                pass

            elif len(self.older_classes) > 0:
                if self.args.lwf:
                    # This is for warm up period; i.e, for the first four epochs, only train the fc layers.
                    if epoch == 0 and batch_idx == 0:
                        for param in self.model.named_parameters():
                            if "fc" in param[0]:
                                param[1].requies_grad = True
                            else:
                                param[1].requires_grad = False
                    if epoch == 4 and batch_idx == 0:
                        for param in self.model.parameters():
                            param.requires_grad = True
                # Get softened targets generated from previous model;
                pred2 = self.model_fixed(Variable(data), T=myT, labels=True)
                # Softened output of the model
                output2 = self.model(Variable(data), T=myT)
                # Compute second loss
                loss2 = F.kl_div(output2, Variable(pred2.data))
                # Store the gradients in the gradient buffers
                loss2.backward(retain_graph=True)
                # Scale the stored gradients by a factor of my
                for param in self.model.parameters():
                    param.grad=param.grad*(myT*myT)*self.args.alpha
            loss.backward()
            self.optimizer.step()

import os
import cv2

from torch.optim import SGD

from model.misc_functions import preprocess_image, recreate_image, get_params

class DisguisedFoolingSampleGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent, breaks as soon as
        the target prediction confidence is captured
    """

    def __init__(self, model, minimum_confidence, cuda, iterator):
        self.iterator = iterator
        self.model = model
        self.model.eval()
        self.cuda = cuda

    def generate(self):
        self.iterator.dataset.no_transformation = True
        for batch_idx, (data, target) in enumerate(self.iterator):

            ut.visualizeTensor(data.cpu(), "../pathData.jpg")
            if self.cuda:
                data = data.cuda()
                target = target.cuda()
            outputTemp = self.model(Variable(data), getAllFeatures=True).data
            # self.processed_images = Variable(data, requires_grad=True)

            # self.processed_image = Variable(self.initial_image*2, requires_grad=True)
            instance = torch.mean(data,dim=0)
            instance = instance.unsqueeze(0).repeat(100,1,1,1)
            perm = torch.randperm(100)
            if self.cuda:
                perm = perm.cuda()
            instance = data[perm]
            instance = data
            # print ("shape of instance", instance.shape)
            # print ("Shape of input", self.processed_images.shape)
            if self.cuda:
                instance = instance.cuda()
            self.processed_image = Variable(instance, requires_grad=True)
            lRate = 0.000001
            optimizer = SGD([self.processed_image], lr=lRate, momentum=0.9)

            for i in range(1, 300):
                # Process image and return variable
                # self.processed_image = preprocess_image(self.initial_image)
                # Define optimizer for the image
                if i == 150:
                    lRate/=10
                if i == 250:
                    lRate/=10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lRate
                # Forward

                output = self.model(self.processed_image, getAllFeatures=True)
                # Get confidence from softmax


                # Target specific class
                class_loss = F.mse_loss(output, Variable(outputTemp),reduce=False)

                # class_loss = -output[0, self.target_class]
                print('Iteration:', str(i), 'Class Loss', class_loss)
                # Zero grads
                self.processed_image.grad = None
                # Backward
                class_loss.backward(outputTemp)
                # Update image
                optimizer.step()
                # Recreate image
                # self.initial_image = recreate_image(self.processed_image)
                # Save image
                # Set min = 0 and max = 1
                # self.processed_image.data = self.processed_image.data - torch.mean(self.processed_image.data)
                self.processed_image.data = self.processed_image.data - torch.min(self.processed_image.data) + 0.2
                self.processed_image.data = self.processed_image.data/torch.max(self.processed_image.data)
                if i%100 == 1:
                    ut.visualizeTensor(self.processed_image.data.cpu(), "../path"+str(i)+".jpg")
            # tempData  = torchvision.transforms.ToPILImage()(self.processed_image.data.cpu().numpy())
            tempData = np.swapaxes(np.swapaxes(self.processed_image.data.cpu().numpy(),1,3), 1,2)
            self.iterator.dataset.data[target.cpu().numpy()] = tempData*255
            for batch_idx, (data, target) in enumerate(self.iterator):
                ut.visualizeTensor(data.cpu(), "../pathDataTemp.jpg")
                break

        self.iterator.dataset.no_transformation = True
        return self.processed_image.data
