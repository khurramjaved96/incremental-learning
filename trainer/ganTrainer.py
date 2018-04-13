import os
import copy
import time
import itertools

import torch
import numpy as np
import torch.nn as nn
import utils.utils as ut
import torch.optim as optim
import torch.utils.data as td
import matplotlib.pyplot as plt
from torch.autograd import Variable

import trainer.classifierTrainer as t
import trainer.classifierFactory as tF
import trainer.gans.gutils as gutils
from trainer.gans.ganFactory import GANFactory

class Trainer():
    def __init__(
            self, args, dataset, classifier_trainer, model, train_iterator,
            test_iterator, train_loader, model_factory, experiment,
            train_iterator_ideal, train_loader_ideal):
        '''
        ideal params are for ideal nmc calculation
        '''
        self.args = args
        self.batch_size = args.batch_size
        self.dataset = dataset
        self.classifier_trainer = classifier_trainer
        self.model = model
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.train_loader = train_loader
        self.model_factory = model_factory
        self.experiment = experiment
        self.old_classes = None
        self.G = None
        self.D = None
        self.fixed_g = None
        self.examples = {}
        self.increment = 0
        self.fixed_noise = torch.randn(100,100,1,1)
        self.is_cond = args.process == "cdcgan"
        if args.ideal_nmc:
            self.train_iterator_ideal = train_iterator_ideal
            self.train_loader_ideal = train_loader_ideal
        if args.dataset == "MNIST" or args.dataset == "CIFAR10":
            self.num_classes = 10
        else:
            self.num_classes = 100
        # Do not use this for training, it is volatile
        if args.cuda:
            self.fixed_noise  = Variable(self.fixed_noise.cuda(), volatile=True)



    def train(self):
        x = []
        y = []
        y_nmc = []
        y_nmc_ideal = []

        #Get NMC and NMC ideal (if required)
        test_factory = tF.ClassifierFactory()
        nmc = test_factory.get_tester("nmc", self.args.cuda)
        if self.args.ideal_nmc:
            ideal_nmc = test_factory.get_tester("nmc", self.args.cuda)

        #Get the appropriate GAN trainer
        gan_trainer = GANFactory.get_trainer(self.args.process,
                                             self.args,
                                             self.num_classes,
                                             self.train_iterator,
                                             self.classifier_trainer.model_fixed,
                                             self.experiment)

        for class_group in range(0, self.dataset.classes, self.args.step_size):
            #Setup training and increment classes
            self.classifier_trainer.setup_training()
            self.classifier_trainer.increment_classes(class_group)
            #If not first increment, then generate examples and replace data
            if class_group > 0:
                self.increment = self.increment + 1
                self.old_classes = self.classifier_trainer.older_classes
                self.examples = gutils.generate_examples(self.args,
                                                         self.fixed_g,
                                                         self.args.gan_num_examples,
                                                         self.old_classes,
                                                         self.num_classes,
                                                         self.fixed_noise,
                                                         self.experiment,
                                                         "Final-Inc"+str(self.increment-1),
                                                         True, self.is_cond)
                #TODO put trainLoader
                self.train_iterator.dataset.replace_data(self.examples,
                                                         self.args.gan_num_examples)
                # Send examples to CPU
                if self.is_cond:
                    for k in self.examples:
                        self.examples[k] = self.examples[k].data.cpu()

            #-------------Train Classifier-----------#
            epoch = 0
            for epoch in range(0, self.args.epochs_class):
                self.classifier_trainer.update_lr(epoch)
                self.classifier_trainer.train(self.examples, self.old_classes,
                                             self.batch_size)
                if epoch % self.args.log_interval == 0:
                    print("[Classifier] Train:",
                          self.classifier_trainer.evaluate(self.train_iterator),
                          "Test:",
                          self.classifier_trainer.evaluate(self.test_iterator))

            self.classifier_trainer.update_frozen_model()

            #-------------Using NMC Classifier-----------#
            nmc.update_means(self.model, self.train_iterator, self.args.cuda,
                            self.dataset.classes, self.old_classes, self.is_cond)
            nmc_train = nmc.classify(self.model, self.train_iterator,
                                        self.args.cuda, True)
            nmc_test = nmc.classify(self.model, self.test_iterator,
                                    self.args.cuda, True)
            y_nmc.append(nmc_test)

            if self.args.ideal_nmc:
                ideal_nmc.update_means(self.model, self.train_iterator_ideal, self.args.cuda,
                                      self.dataset.classes, [], True)
                nmc_test_ideal = ideal_nmc.classify(self.model, self.test_iterator,
                                                    self.args.cuda, True)
                y_nmc_ideal.append(nmc_test_ideal)

            print("Train NMC: ", nmc_train)
            print("Test NMC: ", nmc_test)
            if self.args.ideal_nmc:
                print("Test NMC (Ideal)", nmc_test_ideal)

            #-------------Train GAN-----------#
            #Get new G and D only if it doesn't exist and persist_gan is off
            if self.G == None or not self.args.persist_gan:
                self.G, self.D = self.model_factory.get_model(self.args.process,
                                                              self.args.dataset,
                                                              self.args.minibatch_discrimination,
                                                              self.args.gan_d)
                if self.args.cuda:
                    self.G = self.G.cuda()
                    self.D = self.D.cuda()

            #Load if we should be loading from ckpt, otherwise train
            is_loaded = False
            if self.args.load_g_ckpt != '':
                is_loaded = gutils.load_checkpoint(self.args.load_g_ckpt,
                                                   self.increment,
                                                   self.G)
            #Train the GAN, use alternative transform
            if not is_loaded:
                self.train_loader.do_alt_transform = True
                gan_trainer.train(self.G,
                                  self.D,
                                  self.train_iterator.dataset.active_classes,
                                  self.increment)
                self.train_loader.do_alt_transform = False

            #Optimize features, default off
            if self.args.optimize_features:
                self.optimize_features()
            self.update_frozen_generator()

            #Save ckpt if required
            if self.args.save_g_ckpt:
                gutils.save_checkpoint(self.args.gan_epochs[self.increment],
                                       self.increment, self.experiment, self.G)

            #-------------Save and Plot data-----------#
            #Saving confusion matrix
            ut.saveConfusionMatrix(int(class_group / self.args.step_size) *
                                   self.args.epochs_class + epoch,
                                   self.experiment.path + "CONFUSION",
                                   self.model, self.args, self.dataset,
                                   self.test_iterator)

            #Plot
            y.append(self.classifier_trainer.evaluate(self.test_iterator))
            x.append(class_group + self.args.step_size)
            results = [("Trained Classifier",y), ("NMC Classifier", y_nmc)]
            if self.args.ideal_nmc:
                results.append(("Ideal NMC Classifier", y_nmc_ideal))
                ut.plotEmbeddings(self.experiment,
                                  [("NMC_means", nmc.means ), ("Ideal_NMC_means", ideal_nmc.means)],
                                   "Inc"+str(self.increment))
            ut.plotAccuracy(self.experiment, x,
                            results,
                            self.dataset.classes + 1, self.args.name)

    def update_frozen_generator(self):
        self.G.eval()
        self.fixed_g = copy.deepcopy(self.G)
        for param in self.fixed_g.parameters():
            param.requires_grad = False

    def unfreeze_frozen_generator(self):
        self.G.train()
        for param in self.G.parameters():
            param.requires_grad = True

    def optimize_features(self):
        '''
        Attempts to reduce the Euclidean distance between
        the batches of features of generated and real images
        '''
        self.unfreeze_frozen_generator()
        model = self.classifier_trainer.model_fixed
        optimizer = optim.Adam(self.G.parameters(), lr=self.args.optimize_feat_lr, betas=(0.5, 0.999))
        euclidean_dist = nn.PairwiseDistance(2)
        print("Optimizing features")
        for epoch in range(self.args.optimize_feat_epochs):
            losses = []
            start_time = time.time()
            for batch_idx, (image, label) in enumerate(self.train_iterator):
                batch_size = image.shape[0]
                # Generate noise
                g_random_noise = torch.randn((batch_size, 100))
                g_random_noise = g_random_noise.view(-1, 100, 1, 1)
                #TODO wrap in variable for noncuda
                if self.args.cuda:
                    image = Variable(image.cuda())
                    g_random_noise  = Variable(g_random_noise.cuda())
                # Generate examples
                g_output = self.G(g_random_noise)
                # Generate features for real and fake images
                output_fake = model.forward(g_output, True)
                output_real = model.forward(image, True)
                # Calculate euclidean distance
                loss = torch.mean(euclidean_dist(output_fake, output_real))
                loss.backward()
                optimizer.step()
                losses.append(loss)

            # Calculate mean loss, save examples and print stats
            mean_loss = (sum(losses)/len(losses)).cpu().data.numpy()[0]
            if epoch % self.args.gan_img_save_interval == 0:
                self.generate_examples(self.G, 100, self.train_iterator.dataset.active_classes,
                                      "OPT-Inc"+str(self.increment) +
                                      "_E" + str(epoch), True)
            print("[GAN-OPTIMIZE] Epoch:", epoch,
                  "Loss:", mean_loss,
                  "Time taken:", time.time() - start_time)
        self.update_frozen_generator()

