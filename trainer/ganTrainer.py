import os
import copy
import torch
import itertools
import numpy as np
import torch.nn as nn
import utils.utils as ut
import torch.optim as optim
import torch.utils.data as td
import matplotlib.pyplot as plt
import trainer.classifierTrainer as t
import trainer.classifierFactory as tF
from torch.autograd import Variable

class trainer():
    def __init__(self, args, dataset, classifierTrainer, model, trainIterator,
                 testIterator, trainLoader, modelFactory, experiment):
        self.args = args
        self.batch_size = args.batch_size
        self.dataset = dataset
        self.classifierTrainer = classifierTrainer
        self.model = model
        self.trainIterator = trainIterator
        self.testIterator = testIterator
        self.trainLoader = trainLoader
        self.modelFactory = modelFactory
        self.experiment = experiment
        self.old_classes = None
        self.G = None
        self.D = None
        self.fixed_G = None
        self.examples = {}
        self.labels = {}
        self.increment = 0
        self.is_C = args.process == "cgan"

    def train(self):
        x = []
        y = []

        for classGroup in range(0, self.dataset.classes, self.args.step_size):
            self.classifierTrainer.setupTraining()
            self.classifierTrainer.incrementClasses(classGroup)
            #Get new iterator with reduced batch_size
            if classGroup > 0:
                self.increment = self.increment + 1
                self.old_classes = self.classifierTrainer.olderClasses
                #Generate examples
                self.examples, self.labels = self.generateExamples(self.fixed_G,
                                                                   self.args.gan_num_examples,
                                                                   self.old_classes,
                                                                   "Final-Inc"+str(self.increment-1),
                                                                   True)
                if not self.is_C:
                    print("replaceData is not handling standard GAN yet")
                    assert False
                self.trainIterator.dataset.replaceData(self.examples, self.args.gan_num_examples)
                #This is done because the insert_generated_examples is given cpu data
                #TODO See if we can keep this on GPU
                if self.is_C:
                    for k in self.examples:
                        self.examples[k] = self.examples[k].data.cpu()
            epoch = 0
            for epoch in range(0, self.args.epochs_class):
                self.classifierTrainer.updateLR(epoch)
                self.classifierTrainer.train(self.examples, self.old_classes,
                                             self.batch_size)
                if epoch % self.args.log_interval == 0:
                    print("[Classifier] Train:",
                          self.classifierTrainer.evaluate(self.trainIterator),
                          "Test:",
                          self.classifierTrainer.evaluate(self.testIterator))

            self.classifierTrainer.updateFrozenModel()
            #Get a new Generator and Discriminator
            if self.G == None or not self.args.persist_gan:
                if self.args.process == "cgan":
                    self.G, self.D = self.modelFactory.getModel("cdcgan", self.args.dataset)
                else:
                    self.G, self.D = self.modelFactory.getModel("dcgan", self.args.dataset)
                if self.args.cuda:
                    self.G = self.G.cuda()
                    self.D = self.D.cuda()
            self.trainGan(self.G, self.D, self.is_C)
            self.updateFrozenGenerator(self.G)

            # Saving confusion matrix
            ut.saveConfusionMatrix(int(classGroup / self.args.step_size) *
                                   self.args.epochs_class + epoch,
                                   self.experiment.path + "CONFUSION",
                                   self.model, self.args, self.dataset,
                                   self.testIterator)

            y.append(self.classifierTrainer.evaluate(self.testIterator))
            x.append(classGroup + self.args.step_size)
            ut.plotAccuracy(self.experiment, x,
                            [("Trained Classifier",y)],
                            self.dataset.classes + 1, self.args.name)

    def trainGan(self, G, D, is_C):
        activeClasses = self.trainIterator.dataset.activeClasses
        print("ACTIVE CLASSES: ", activeClasses)

        #TODO Change batchsize of dataIterator here to gan_batch_size
        criterion = nn.BCELoss()
        G_Opt = optim.Adam(G.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))
        D_Opt = optim.Adam(D.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))

        #Matrix of shape [10,10,1,1] with 1s at positions
        #where shape[0]==shape[1]
        if is_C:
            GVec = torch.zeros(10, 10)
            GVec = GVec.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

            #Matrix of shape [10,10,32,32] with 32x32 matrix of 1s
            #where shape[0]==shape[1]
            DVec = torch.zeros([10, 10, 32, 32])
            for i in range(10):
                DVec[i, i, :, :] = 1

        print("Starting GAN Training")
        for epoch in range(int(self.args.gan_epochs[self.increment])):
            G.train() #Remove the one from generate_examples too if this is removed
            D_Losses = []
            G_Losses = []
            self.updateLR(epoch, G_Opt, D_Opt)

            #Iterate over examples that the classifier trainer just iterated on
            for image, label in self.trainIterator:
                batch_size = image.shape[0]
                if self.increment > 0:
                    self.saveResults(image, "sample", is_tensor=True, axis_size=11)

                #Make vectors of ones and zeros of same shape as output by
                #Discriminator so that it can be used in BCELoss
                D_like_real = torch.ones(batch_size)
                D_like_fake = torch.zeros(batch_size)
                if self.args.cuda:
                    D_like_real = Variable(D_like_real.cuda())
                    D_like_fake = Variable(D_like_fake.cuda())
                ##################################
                #Train Discriminator
                ##################################
                #Train with real image and labels
                D.zero_grad()

                #Shape [batch_size, 10, 32, 32]. Each entry at D_labels[0]
                #contains 32x32 matrix of 1s inside D_labels[label] index
                #and 32x32 matrix of 0s otherwise
                D_labels = DVec[label] if is_C else None
                if self.args.cuda:
                    image    = Variable(image.cuda())
                    D_labels = Variable(D_labels.cuda()) if is_C else None

                #Discriminator output for real image and labels
                D_output = D(image, D_labels).squeeze() if is_C else D(image).squeeze()
                #Maximize the probability of D_output to be all 1s
                D_real_loss = criterion(D_output, D_like_real)

                #Train with fake image and labels
                G_random_noise = torch.randn((batch_size, 100))
                G_random_noise = G_random_noise.view(-1, 100, 1, 1)

                if is_C:
                    #Generating random batch_size of labels from amongst
                    #labels present in activeClass
                    random_labels = torch.from_numpy(np.random.choice(activeClasses,
                                                                      batch_size))
                    #Convert labels to appropriate shapes
                    G_random_labels = GVec[random_labels]
                    D_random_labels = DVec[random_labels]

                if self.args.cuda:
                    G_random_noise  = Variable(G_random_noise.cuda())
                    G_random_labels = Variable(G_random_labels.cuda()) if is_C else None
                    D_random_labels = Variable(D_random_labels.cuda()) if is_C else None

                G_output = G(G_random_noise, G_random_labels) if is_C else G(G_random_noise)
                D_output = D(G_output, D_random_labels).squeeze() if is_C else D(G_output).squeeze()

                D_fake_loss = criterion(D_output, D_like_fake)
                D_Loss = D_real_loss + D_fake_loss
                D_Losses.append(D_Loss)
                D_Loss.backward()
                D_Opt.step()

                #################################
                #Train Generator
                #################################
                G.zero_grad()
                #Follow same steps, but change the loss
                G_random_noise = torch.randn((batch_size, 100))
                G_random_noise = G_random_noise.view(-1, 100, 1, 1)

                if is_C:
                    random_labels = torch.from_numpy(np.random.choice(activeClasses,
                                                                      batch_size))
                    #Convert labels to appropriate shapes
                    G_random_labels = GVec[random_labels]
                    D_random_labels = DVec[random_labels]

                if self.args.cuda:
                    G_random_noise  = Variable(G_random_noise.cuda())
                    G_random_labels = Variable(G_random_labels.cuda()) if is_C else None
                    D_random_labels = Variable(D_random_labels.cuda()) if is_C else None

                G_output = G(G_random_noise, G_random_labels) if is_C else G(G_random_noise)
                D_output = D(G_output, D_random_labels).squeeze() if is_C else D(G_output).squeeze()

                G_Loss = criterion(D_output, D_like_real)
                G_Loss.backward()
                G_Losses.append(G_Loss)
                G_Opt.step()

            print("[GAN] Epoch:", epoch,
                  "G_Loss:", (sum(G_Losses)/len(G_Losses)).cpu().data.numpy()[0],
                  "D_Loss:", (sum(D_Losses)/len(D_Losses)).cpu().data.numpy()[0])
            self.generateExamples(G, 100, activeClasses,
                                  "Inc"+str(self.increment) + "_E" + str(epoch), True)

    #Uses GAN to generate examples
    def generateExamples(self, G, num_examples, active_classes, name="", save=False):
        G.eval()
        if self.is_C:
            examples = {}
            labels = {}
            for klass in active_classes:
                for _ in range(num_examples//100):
                    noise = torch.randn(100,100,1,1)
                    targets = torch.zeros(100,10,1,1)
                    targets[:, klass] = 1
                    if self.args.cuda:
                        noise  = Variable(noise.cuda(), volatile=True)
                        targets = Variable(targets.cuda(), volatile=True)
                    images = G(noise, targets)
                    if not klass in examples.keys():
                        labels[klass] = targets
                        examples[klass] = images
                    else:
                        labels[klass] = torch.cat((labels[klass], targets), dim=0)
                        examples[klass] = torch.cat((examples[klass],images), dim=0)
                if save:
                    self.saveResults(examples[klass][0:100], name + "_C" + str(klass))
        else:
            examples = []
            labels = None
            for i in range(num_examples//100):
                noise = torch.randn(100,100,1,1)
                if self.args.cuda:
                    noise  = Variable(noise.cuda(), volatile=True)
                images = G(noise)
                if len(examples) == 0:
                    examples = images
                else:
                    examples = torch.cat((examples, images), dim=0)
            if save:
                self.saveResults(examples[0:100], name + "_smpl" + str(i))
        return examples, labels

    def updateFrozenGenerator(self, G):
        G.eval()
        self.fixed_G = copy.deepcopy(G)
        for param in self.fixed_G.parameters():
            param.requires_grad = False

    def saveResults(self, images, name, is_tensor=False, axis_size=10):
        _, sub = plt.subplots(axis_size, axis_size, figsize=(5, 5))
        for i, j in itertools.product(range(axis_size), range(axis_size)):
            sub[i, j].get_xaxis().set_visible(False)
            sub[i, j].get_yaxis().set_visible(False)

        for k in range(axis_size * axis_size):
            i = k // axis_size
            j = k % axis_size
            sub[i, j].cla()
            if is_tensor:
                sub[i, j].imshow(images[k, 0].cpu().numpy(), cmap='gray')
            else:
                sub[i, j].imshow(images[k, 0].cpu().data.numpy(), cmap='gray')

        plt.savefig(self.experiment.path + "results/" + name + ".png")
        plt.cla()
        plt.clf()
        plt.close()

    def updateLR(self, epoch, G_Opt, D_Opt):
        for temp in range(0, len(self.args.gan_schedule)):
            if self.args.gan_schedule[temp] == epoch:
                #Update Generator LR
                for param_group in G_Opt.param_groups:
                    currentLr_G = param_group['lr']
                    param_group['lr'] = currentLr_G * self.args.gan_gammas[temp]
                    print("Changing GAN Generator learning rate from",
                          currentLr_G, "to", currentLr_G * self.args.gan_gammas[temp])
                #Update Discriminator LR
                for param_group in D_Opt.param_groups:
                    currentLr_D = param_group['lr']
                    param_group['lr'] = currentLr_D * self.args.gan_gammas[temp]
                    print("Changing GAN Generator learning rate from",
                          currentLr_D, "to", currentLr_D * self.args.gan_gammas[temp])
