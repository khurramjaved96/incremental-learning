from __future__ import print_function

import copy

import torch
import torch.nn.functional as F
import torch.utils.data as td
from torch.autograd import Variable
import random

class Trainer():
    def __init__(self, train_data_iterator, test_data_iterator, dataset, model, args, optimizer, train_data_iterator_ideal=None):
        self.train_data_iterator = train_data_iterator
        self.test_data_ierator = test_data_iterator
        self.train_data_iterator_ideal = train_data_iterator_ideal
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
        if args.ideal_nmc:
            self.train_loader_ideal = self.train_data_iterator_ideal.dataset
        if not args.no_random:
            print("Randomly shuffling classes")
            random.seed(args.seed)
            random.shuffle(self.all_classes)

    def update_iterator(self, train_iterator):
        self.train_data_iterator = train_iterator
        self.train_loader = train_iterator.dataset

    def update_lr(self, epoch):
        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from", self.current_lr, "to", self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self, class_group):
        for temp in range(class_group, class_group + self.args.step_size):
            pop_val = self.all_classes.pop()
            self.train_data_iterator.dataset.add_classes(pop_val)
            self.test_data_ierator.dataset.add_classes(pop_val)
            if self.args.ideal_nmc:
                self.train_data_iterator_ideal.dataset.add_classes(pop_val)
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
        if n not in self.older_classes:
            self.older_classes.append(n)

    def setup_training(self):
        for param_group in self.optimizer.param_groups:
            print("Setting LR to", self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr

        k = 0
        if self.args.process == "nmc" and self.left_over != []:
            k = int(self.args.memory_budget / len(self.left_over))
        for val in self.left_over:
            self.limit_class(val, k, not self.args.no_herding)


    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def insert_generated_images(self, data, target, gan_images, gan_labels, batch_size, is_cond=False):
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
                if is_cond:
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

    def train(self, gan_images=None, gan_labels=None, batch_size=None, D=None, epoch=0):
        torch.manual_seed(self.args.seed + epoch)
        self.model.train()
        #TODO CHECK MEMORY
        if D is not None:
            for param in D.parameters():
                param.requires_grad = False
        #    D.eval()

        for batch_idx, (data, target) in enumerate(self.train_data_iterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            weight_vector = (target * 0).int()
            for elem in self.older_classes:
                weight_vector = weight_vector + (target == elem).int()

            old_classes_indices = torch.squeeze(torch.nonzero((weight_vector > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((weight_vector == 0)).long())
            self.optimizer.zero_grad()

            y_onehot = torch.FloatTensor(len(data), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)

            output, output2 = self.model(Variable(data), T=self.args.T, both=True)
            if self.args.ac_distill and D is not None:
                pred2 = D(Variable(data, True), T=self.args.T)[1]
                loss2 = F.kl_div(output2, Variable(pred2.data))
                loss2.backward(retain_graph=True)
                alpha = 1
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad * (self.args.T * self.args.T) * alpha
                # y_onehot = pred2.data
            elif not self.args.no_distill:
                if len(self.older_classes) > 0:
                    pred2 = self.model_fixed(Variable(data), labels=True, T=self.args.T)
                    loss2 = F.kl_div(output2, Variable(pred2.data))
                    loss2.backward(retain_graph=True)
                    alpha=1
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = param.grad * (self.args.T * self.args.T) * alpha
                    # y_onehot[:, self.older_classes] = pred2.data[:, self.older_classes]
                loss = F.kl_div(output, Variable(y_onehot))
                loss.backward()
            self.optimizer.step()

    #TODO Add generated images here
    def evaluate(self, loader, D=None):
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, target in loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            if D is not None:
                output = D(data, T=1)[1]
            else:
                output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(loader.dataset)
        return 100. * correct / len(loader.dataset)
