import torch
import pickle
import itertools
import copy
import torch
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
        self.examples = {}
        self.labels = {}

    def train(self):
        x = []
        y = []

        for classGroup in range(0, self.dataset.classes, self.args.step_size):
            self.classifierTrainer.setupTraining()
            self.classifierTrainer.incrementClasses(classGroup)
            #Get new iterator with reduced batch_size
            if classGroup > 0:
                self.batch_size = self.batch_size // 2
                self.old_classes = self.classifierTrainer.olderClasses
                self.trainIterator = ut.get_new_iterator(self.args.cuda,
                                                         self.trainLoader,
                                                         self.batch_size)
            #Generate examples
                self.examples, self.labels = self.generateExamples(self.G,
                                                                   self.args.gan_num_examples,
                                                                   self.old_classes)
                for k in self.examples:
                    self.examples[k] = ut.normalize_images(self.examples[k]).data.cpu()
                    self.saveResults(self.examples[k], 22, k, True)

            epoch = 0
            for epoch in range(0, self.args.epochs_class):
                self.classifierTrainer.updateLR(epoch)
                self.classifierTrainer.updateIterator(self.trainIterator)
                self.classifierTrainer.train(self.examples, self.old_classes,
                                             self.batch_size)
                if epoch % self.args.log_interval == 0:
                    print("[Classifier] Train:",
                          self.classifierTrainer.evaluate(self.trainIterator),
                          "Test:",
                          self.classifierTrainer.evaluate(self.testIterator))

            #Get a new Generator and Discriminator
            #TODO What if we kept the Discriminator?
            G, D = self.modelFactory.getModel("cdcgan", self.args.dataset)
            if self.args.cuda:
                G.cuda()
                D.cuda()
            self.trainGan(G, D)
            self.updateFrozenGenerator(G)

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

    def trainGan(self, G, D):
        activeClasses = self.trainIterator.dataset.activeClasses

        #TODO Change batchsize of dataIterator here to gan_batch_size
        #dataIterator.batch_size = self.args.gan_batch_size <- Doesnt work :(
        criterion = nn.BCELoss()
        G_Opt = optim.Adam(G.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))
        D_Opt = optim.Adam(D.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))

        #Matrix of shape [10,10,1,1] with 1s at positions
        #where shape[0]==shape[1]
        GVec = torch.zeros(10, 10)
        GVec = GVec.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

        #Matrix of shape [10,10,32,32] with 32x32 matrix of 1s
        #where shape[0]==shape[1]
        DVec = torch.zeros([10, 10, 32, 32])
        for i in range(10):
            DVec[i, i, :, :] = 1

        print("Starting GAN Training")
        for epoch in range(int(self.args.gan_epochs)):
            D_Losses = []
            G_Losses = []
            self.updateLR(epoch, G_Opt, D_Opt)

            #Iterate over examples that the classifier trainer just iterated on
            #TODO Also add examples that are generated by GAN for active classes
            #dataIteratorForGan contains examples only from current increment
            for image, label in self.trainIterator:
                if self.old_classes != None:
                    image, label = self.classifierTrainer.insert_generated_images(
                                   image, label, self.examples, self.old_classes,
                                   self.batch_size)
                batch_size = image.shape[0]

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
                D_labels = DVec[label]
                if self.args.cuda:
                    image    = Variable(image.cuda())
                    D_labels = Variable(D_labels.cuda())

                #Discriminator output for real image and labels
                D_output = D(image, D_labels).squeeze()
                #Maximize the probability of D_output to be all 1s
                D_real_loss = criterion(D_output, D_like_real)

                #Train with fake image and labels
                G_random_noise = torch.randn((batch_size, 100))
                G_random_noise = G_random_noise.view(-1, 100, 1, 1)

                #Generating random batch_size of labels from amongst
                #labels present in activeClass
                random_labels = torch.from_numpy(np.random.choice(activeClasses,
                                                                  batch_size))
                #Convert labels to appropriate shapes
                G_random_labels = GVec[random_labels]
                D_random_labels = DVec[random_labels]

                if self.args.cuda:
                    G_random_noise  = Variable(G_random_noise.cuda())
                    G_random_labels = Variable(G_random_labels.cuda())
                    D_random_labels = Variable(D_random_labels.cuda())

                G_output = G(G_random_noise, G_random_labels)
                D_output = D(G_output, D_random_labels).squeeze()

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
                #TODO put this in a function instead
                G_random_noise = torch.randn((batch_size, 100))
                G_random_noise = G_random_noise.view(-1, 100, 1, 1)

                random_labels = torch.from_numpy(np.random.choice(activeClasses,
                                                                  batch_size))
                #Convert labels to appropriate shapes
                G_random_labels = GVec[random_labels]
                D_random_labels = DVec[random_labels]

                if self.args.cuda:
                    G_random_noise  = Variable(G_random_noise.cuda())
                    G_random_labels = Variable(G_random_labels.cuda())
                    D_random_labels = Variable(D_random_labels.cuda())

                G_output = G(G_random_noise, G_random_labels)
                D_output = D(G_output, D_random_labels).squeeze()

                G_Loss = criterion(D_output, D_like_real)
                G_Loss.backward()
                G_Losses.append(G_Loss)
                G_Opt.step()

            print("[GAN] Epoch:", epoch,
                  "G_Loss:", (sum(G_Losses)/len(G_Losses)).cpu().data.numpy()[0],
                  "D_Loss:", (sum(D_Losses)/len(D_Losses)).cpu().data.numpy()[0])
            self.generateExamples(G, 100, activeClasses, epoch, save=True)
            if self.old_classes != None:
                self.generateExamples(G, 100, activeClasses, epoch + 100, save=True)


    #Uses GAN to generate examples
    def generateExamples(self, G, num_examples, active_classes, epoch=0, save=False):
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
                    G.eval()
                    images = G(noise, targets)
                    G.train()
                    if not klass in examples.keys():
                        labels[klass] = targets
                        examples[klass] = images
                    else:
                        labels[klass] = torch.cat((labels[klass], targets), dim=0)
                        examples[klass] = torch.cat((examples[klass],images), dim=0)
                    if save:
                        self.saveResults(images, epoch, klass)
            return examples, labels

    def updateFrozenGenerator(self, G):
        G.eval()
        self.G = copy.deepcopy(G)
        for param in self.G.parameters():
            param.requires_grad = False

    def saveResults(self, images, epoch, klass, is_tensor=False):
        _, sub = plt.subplots(10, 10, figsize=(5, 5))
        for i, j in itertools.product(range(10), range(10)):
            sub[i, j].get_xaxis().set_visible(False)
            sub[i, j].get_yaxis().set_visible(False)

        for k in range(100):
            i = k // 10
            j = k % 10
            sub[i, j].cla()
            if is_tensor:
                sub[i, j].imshow(images[k, 0].cpu().numpy(), cmap='gray')
            else:
                sub[i, j].imshow(images[k, 0].cpu().data.numpy(), cmap='gray')

        plt.savefig("results/" + str(epoch) + "_" + str(klass) + ".png")

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
