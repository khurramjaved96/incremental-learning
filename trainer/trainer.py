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

        self.old_models = {}
        self.cached_adversarial_instances = {}

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

    def convert_to_adversarial_instance(self, instance, target_class, target_instance, alpha = 0.1, iters = 25):
        # generate adversarial instances only through the least updated model for any class
        if target_class not in self.old_models:
            self.old_models[target_class] = copy.deepcopy(self.model_fixed)
            self.old_models[target_class].eval()

        # retrieve from cache if possible
        if target_class in self.cached_adversarial_instances and len(self.cached_adversarial_instances[target_class]) > 200:
            return random.choice(self.cached_adversarial_instances[target_class])

        instance.unsqueeze_(0)
        target_instance.unsqueeze_(0)

        ce_loss = nn.CrossEntropyLoss()
        im_label_as_var = torch.from_numpy(np.asarray([target_class]))
        im_label_as_var = Variable(im_label_as_var)

        if self.args.cuda:
            instance = instance.cuda()
            target_instance = target_instance.cuda()
            im_label_as_var = im_label_as_var.cuda()
        instance = Variable(instance, requires_grad=True)
        target_instance = Variable(target_instance, requires_grad=False)

        # Store for later use
        outputFeatureTarget, target_confidence = self.old_models[target_class](target_instance, featureWithLabels=True, T=1)
        target_confidence = (target_confidence)[0][target_class].data.cpu().numpy()[0]
        outputFeatureTarget = Variable(outputFeatureTarget.data, requires_grad=False)

        stage1Target, stage2Target, stage3Target, finalLayerTarget = self.old_models[target_class](target_instance, allStages=True)
        stage1Target = Variable(stage1Target.data, requires_grad=False)
        stage1Num = stage1Target.nelement()
        stage2Target = Variable(stage2Target.data, requires_grad=False)
        stage2Num = stage2Target.nelement()
        stage3Target = Variable(stage3Target.data, requires_grad=False)
        stage3Num = stage3Target.nelement()
        finalLayerTarget = Variable(finalLayerTarget.data, requires_grad=False)
        finalLayerNum = finalLayerTarget.nelement()

        self.old_models[target_class].zero_grad()
        prevLoss = 100000
        for i in range(1, iters):
            instance.grad = None

            # Calculate loss through features in different layers
            stage1Current, stage2Current, stage3Current, finalLayerCurrent, current_confidence = self.old_models[target_class](instance, allStagesWithLabels=True)
            stage1Loss = torch.sum(torch.abs(stage1Current - stage1Target)) / stage1Num
            stage2Loss = torch.sum(torch.abs(stage2Current - stage2Target)) / stage2Num
            stage3Loss = torch.sum(torch.abs(stage3Current - stage3Target)) / stage3Num
            finalLayerLoss = torch.sum(torch.abs(finalLayerCurrent - finalLayerTarget)) / finalLayerNum

            featureLossScalar = finalLayerLoss.data.cpu().numpy().tolist()[0]
            stage1LossScalar = stage1Loss.data.cpu().numpy().tolist()[0]
            stage2LossScalar = stage2Loss.data.cpu().numpy().tolist()[0]
            stage3LossScalar = stage3Loss.data.cpu().numpy().tolist()[0]

            # Get confidences
            current_confidence = (current_confidence)[0][target_class].data.cpu().numpy()[0]

            if abs(prevLoss - featureLossScalar - stage1LossScalar - stage2LossScalar - stage3LossScalar) < 0.02:
                break
            prevLoss = featureLossScalar + stage1LossScalar + stage2LossScalar + stage3LossScalar

            # Zero grads
            self.old_models[target_class].zero_grad()

            # Backward
            stage1Loss.backward(retain_graph=True)
            stage2Loss.backward(retain_graph=True)
            stage3Loss.backward()

            # Update instance with adversarial noise
            adv_noise = alpha * torch.sign(instance.grad.data)
            instance.data -= adv_noise

        # Debug
        print('Iteration:', str(i), 'Target Confidence', "{0:.2f}".format(target_confidence),
              'Current Confidence', "{0:.2f}".format(current_confidence), 'Target ' + str(target_class),
              'Loss ' + str(featureLossScalar + stage1LossScalar + stage2LossScalar + stage3LossScalar))

        # put in cache before returning the generated instance
        result = torch.from_numpy(instance.data.cpu().numpy().squeeze(0)).float()
        if target_class not in self.cached_adversarial_instances:
            self.cached_adversarial_instances[target_class] = []
        self.cached_adversarial_instances[target_class].append(result)

        return result

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
        if k not in self.older_classes:
            self.older_classes.append(k)

    def limit_class(self, n, k, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            # print("Sorting by herding")
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)
        if n not in self.older_classes:
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
            print("minibatch# " + str(batch_idx))
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            weight_vector = (target * 0).int()
            for elem in self.older_classes:
                elem = self.train_loader.indexMapper[elem]
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
                        data[old] = self.convert_to_adversarial_instance(self.dataset.get_random_instance(), target[old], data[old])

            y_onehot = torch.FloatTensor(len(target), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)

            output = self.model(Variable(data), labels=True)
            if not self.args.no_distill:
                older_classes2 = []
                for elem in self.older_classes:
                    older_classes2.append(self.train_loader.indexMapper[elem])
                if len(older_classes2) > 0:
                    pred2 = self.model_fixed(Variable(data), labels=True)
                    y_onehot[:, older_classes2] = pred2.data[:, older_classes2]

            loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss.backward()

            # Freeze any model layers if required
            if self.args.store_features:
                for child in self.model.named_children():
                    if "conv_1_3x3" in child[0] and len(self.older_classes) > 0:
                        for param in child[1].parameters():
                            param.grad *= 0
                    elif "bn" in child[0] and len(self.older_classes) > 0:
                        child[1].eval()
                        for param in child[1].parameters():
                            param.grad *= 0

            self.optimizer.step()
