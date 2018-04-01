import os
import copy
import time
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
                 testIterator, trainLoader, modelFactory, experiment, trainIteratorIdeal, trainLoaderIdeal):
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
        self.increment = 0
        self.fixed_noise = torch.randn(100,100,1,1)
        self.is_C = args.process == "cdcgan"

        if args.ideal_nmc:
            self.trainIteratorIdeal = trainIteratorIdeal
            self.trainLoaderIdeal = trainLoaderIdeal
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

        testFactory = tF.classifierFactory()
        nmc = testFactory.getTester("nmc", self.args.cuda)
        if self.args.ideal_nmc:
            ideal_nmc = testFactory.getTester("nmc", self.args.cuda)

        for classGroup in range(0, self.dataset.classes, self.args.step_size):
            self.classifierTrainer.setupTraining()
            self.classifierTrainer.incrementClasses(classGroup)
            #Get new iterator with reduced batch_size
            if classGroup > 0:
                self.increment = self.increment + 1
                self.old_classes = self.classifierTrainer.olderClasses
                self.examples = self.generateExamples(self.fixed_G,
                                                      self.args.gan_num_examples,
                                                      self.old_classes,
                                                      "Final-Inc"+str(self.increment-1),
                                                      True)
                #TODO put trainLoader
                self.trainIterator.dataset.replaceData(self.examples,
                                                       self.args.gan_num_examples)
                # Send examples to CPU
                if self.is_C:
                    for k in self.examples:
                        self.examples[k] = self.examples[k].data.cpu()

            ######################
            # Train Classifier
            ######################
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

            # Using NMC classifier
            nmc.updateMeans(self.model, self.trainIterator, self.args.cuda,
                            self.dataset.classes, self.old_classes, self.is_C)
            nmc_train = nmc.classify(self.model, self.trainIterator,
                                        self.args.cuda, True)
            nmc_test = nmc.classify(self.model, self.testIterator,
                                    self.args.cuda, True)
            y_nmc.append(nmc_test)

            if self.args.ideal_nmc:
                ideal_nmc.updateMeans(self.model, self.trainIteratorIdeal, self.args.cuda,
                                      self.dataset.classes, [], True)
                nmc_test_ideal = ideal_nmc.classify(self.model, self.testIterator,
                                                    self.args.cuda, True)
                y_nmc_ideal.append(nmc_test_ideal)

            print("Train NMC: ", nmc_train)
            print("Test NMC: ", nmc_test)
            if self.args.ideal_nmc:
                print("Test NMC (Ideal)", nmc_test_ideal)

            #####################
            # Train GAN
            ####################
            if self.G == None or not self.args.persist_gan:
                self.G, self.D = self.modelFactory.getModel(self.args.process,
                                                            self.args.dataset,
                                                            self.args.minibatch_discrimination,
                                                            self.args.gan_d)
                if self.args.cuda:
                    self.G = self.G.cuda()
                    self.D = self.D.cuda()
            is_loaded = False
            if self.args.load_g_ckpt != '':
                is_loaded = self.loadCheckpoint(self.increment)
            if not is_loaded:
                self.trainGan(self.G, self.D, self.is_C, self.num_classes)
            if self.args.optimize_features:
                self.optimizeFeatures()
            self.updateFrozenGenerator()
            if self.args.save_g_ckpt:
                self.saveCheckpoint(self.args.gan_epochs[self.increment])

            # Saving confusion matrix
            ut.saveConfusionMatrix(int(classGroup / self.args.step_size) *
                                   self.args.epochs_class + epoch,
                                   self.experiment.path + "CONFUSION",
                                   self.model, self.args, self.dataset,
                                   self.testIterator)

            # Plot
            y.append(self.classifierTrainer.evaluate(self.testIterator))
            x.append(classGroup + self.args.step_size)
            results = [("Trained Classifier",y), ("NMC Classifier", y_nmc)]
            if self.args.ideal_nmc:
                results.append(("Ideal NMC Classifier", y_nmc_ideal))
                ut.plotEmbeddings(self.experiment,
                                  [("NMC_means", nmc.means ), ("Ideal_NMC_means", ideal_nmc.means)],
                                   "Inc"+str(self.increment))
            ut.plotAccuracy(self.experiment, x,
                            results,
                            self.dataset.classes + 1, self.args.name)

    def optimizeFeatures(self):
        self.unfreezeFrozenGenerator()
        model = self.classifierTrainer.modelFixed
        optimizer = optim.Adam(self.G.parameters(), lr=self.args.optimize_feat_lr, betas=(0.5, 0.999))
        euclidean_dist = nn.PairwiseDistance(2)
        print("Optimizing features")
        for epoch in range(self.args.optimize_feat_epochs):
            losses = []
            startTime = time.time()
            for batch_idx, (image, label) in enumerate(self.trainIterator):
                batch_size = image.shape[0]
                # Generate noise
                G_random_noise = torch.randn((batch_size, 100))
                G_random_noise = G_random_noise.view(-1, 100, 1, 1)
                #TODO wrap in variable for noncuda
                if self.args.cuda:
                    image = Variable(image.cuda())
                    G_random_noise  = Variable(G_random_noise.cuda())
                # Generate examples
                G_output = self.G(G_random_noise)
                # Generate features for real and fake images
                output_fake = model.forward(G_output, True)
                output_real = model.forward(image, True)
                # Calculate euclidean distance
                loss = torch.mean(euclidean_dist(output_fake, output_real))
                loss.backward()
                optimizer.step()
                losses.append(loss)

            # Calculate mean loss, save examples and print stats
            mean_loss = (sum(losses)/len(losses)).cpu().data.numpy()[0]
            if epoch % self.args.gan_img_save_interval == 0:
                self.generateExamples(self.G, 100, self.trainIterator.dataset.activeClasses,
                                      "OPT-Inc"+str(self.increment) +
                                      "_E" + str(epoch), True)
            print("[GAN-OPTIMIZE] Epoch:", epoch,
                  "Loss:", mean_loss,
                  "Time taken:", time.time() - startTime)


    def trainGan(self, G, D, is_C, K):
        G_Losses = []
        D_Losses = []
        activeClasses = self.trainIterator.dataset.activeClasses
        print("ACTIVE CLASSES: ", activeClasses)

        #TODO Change batchsize of dataIterator here to gan_batch_size
        if self.args.process == "wgan":
            if self.args.gan_lr > 5e-5 or len(self.args.gan_schedule) > 1:
                print(">>> NOTICE: Did you mean to set GAN lr/schedule to this value?")
            G_Opt = optim.RMSprop(G.parameters(), lr=self.args.gan_lr)
            D_Opt = optim.RMSprop(D.parameters(), lr=self.args.gan_lr)
        elif self.args.process == "dcgan" or self.args.process == "cdcgan":
            criterion = nn.BCELoss()
            G_Opt = optim.Adam(G.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))
            D_Opt = optim.Adam(D.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))

        #Matrix of shape [K,K,1,1] with 1s at positions
        #where shape[0]==shape[1]
        if is_C:
            tensor = []
            GVec = torch.zeros(K, K)
            for i in range(K):
                tensor.append(i)
            GVec = GVec.scatter_(1, torch.LongTensor(tensor).view(K,1),
                                 1).view(K, K, 1, 1)
            #Matrix of shape [K,K,32,32] with 32x32 matrix of 1s
            #where shape[0]==shape[1]
            DVec = torch.zeros([K, K, 32, 32])
            for i in range(K):
                DVec[i, i, :, :] = 1

        one_sample_saved = False
        a = 0
        b = 0
        print("Starting GAN Training")
        for epoch in range(int(self.args.gan_epochs[self.increment])):
            #######################
            #Start Epoch
            #######################
            G.train()
            D_Losses_E = []
            G_Losses_E = []
            startTime = time.time()
            self.updateLR(epoch, G_Opt, D_Opt)

            #Iterate over examples that the classifier trainer just iterated on
            for batch_idx, (image, label) in enumerate(self.trainIterator):
                batch_size = image.shape[0]
                if not one_sample_saved:
                    self.saveResults(image, "sample_E" + str(epoch), True, np.sqrt(self.args.batch_size))
                    one_sample_saved = True

                #Make vectors of ones and zeros of same shape as output by
                #Discriminator so that it can be used in BCELoss
                if self.args.process == "dcgan" or self.args.process == "cdcgan":
                    smoothing_val = 0
                    if self.args.label_smoothing:
                        smoothing_val = 0.1
                    D_like_real = torch.ones(batch_size) - smoothing_val
                    D_like_fake = torch.zeros(batch_size)
                    if self.args.cuda:
                        D_like_real = Variable(D_like_real.cuda())
                        D_like_fake = Variable(D_like_fake.cuda())
                ##################################
                #Train Discriminator
                ##################################
                #Train with real image and labels
                D.zero_grad()
                a = a + 1

                #Shape [batch_size, 10, 32, 32]. Each entry at D_labels[0]
                #contains 32x32 matrix of 1s inside D_labels[label] index
                #and 32x32 matrix of 0s otherwise
                D_labels = DVec[label] if is_C else None
                if self.args.cuda:
                    image    = Variable(image.cuda())
                    D_labels = Variable(D_labels.cuda()) if is_C else None

                #Discriminator output for real image and labels
                D_output_real = D(image, D_labels).squeeze() if is_C else D(image).squeeze()

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
                G_output = G_output.detach()
                D_output_fake = D(G_output, D_random_labels).squeeze() if is_C else D(G_output).squeeze()

                if self.args.process == "wgan":
                    D_Loss = -(torch.mean(D_output_real) - torch.mean(D_output_fake))

                elif self.args.process == "dcgan" or self.args.process == "cdcgan":
                    D_real_loss = criterion(D_output_real, D_like_real)
                    D_fake_loss = criterion(D_output_fake, D_like_fake)
                    D_Loss = D_real_loss + D_fake_loss

                D_Losses_E.append(D_Loss)
                D_Loss.backward()
                D_Opt.step()

                if self.args.process == "wgan":
                    for p in D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                #################################
                #Train Generator
                #################################
                #Train discriminator more in case of WGAN because the
                #critic needs to be trained to optimality
                if batch_idx % self.args.d_iter != 0:
                    continue

                b = b + 1
                G.zero_grad()
                G_random_noise = torch.randn((batch_size, 100))
                G_random_noise = G_random_noise.view(-1, 100, 1, 1)

                if is_C:
                    random_labels = torch.from_numpy(np.random.choice(activeClasses,
                                                                      batch_size))
                    G_random_labels = GVec[random_labels]
                    D_random_labels = DVec[random_labels]

                if self.args.cuda:
                    G_random_noise  = Variable(G_random_noise.cuda())
                    G_random_labels = Variable(G_random_labels.cuda()) if is_C else None
                    D_random_labels = Variable(D_random_labels.cuda()) if is_C else None

                G_output = G(G_random_noise, G_random_labels) if is_C else G(G_random_noise)
                D_output = D(G_output, D_random_labels).squeeze() if is_C else D(G_output).squeeze()

                if self.args.process == "wgan":
                    G_Loss = -torch.mean(D_output)
                elif self.args.process == "dcgan" or self.args.process == "cdcgan":
                    G_Loss = criterion(D_output, D_like_real)

                G_Loss.backward()
                G_Losses_E.append(G_Loss)
                G_Opt.step()

            #############################
            #End epoch
            #Print Stats and save results
            #############################
            mean_G = (sum(G_Losses_E)/len(G_Losses_E)).cpu().data.numpy()[0]
            mean_D = (sum(D_Losses_E)/len(D_Losses_E)).cpu().data.numpy()[0]
            G_Losses.append(mean_G)
            D_Losses.append(mean_D)

            if epoch % self.args.gan_img_save_interval == 0:
                self.generateExamples(G, 100, activeClasses,
                                      "Inc"+str(self.increment) +
                                      "_E" + str(epoch), True)
                self.saveGANLosses(G_Losses, D_Losses)

            if self.args.save_g_ckpt and epoch % self.args.ckpt_interval == 0:
                self.saveCheckpoint(epoch)
            print("[GAN] Epoch:", epoch,
                  "G_iters:", b,
                  "D_iters:", a,
                  "G_Loss:", mean_G,
                  "D_Loss:", mean_D,
                  "Time taken:", time.time() - startTime)

    def generateExamples(self, G, num_examples, active_classes, name="", save=False):
        '''
        Returns a dict[class] of generated samples.
        In case of Non-Conditional GAN, the samples in the dict are random, they do
        not correspond to the keys in the dict
        Just passing in random noise to the generator and storing the results in dict
        '''
        G.eval()
        examples = {}
        for idx, klass in enumerate(active_classes):
            # Generator outputs 100 images at a time
            for _ in range(num_examples//100):
                #TODO refactor these conditionals
                # Check for memory leak after refactoring
                if self.is_C:
                    targets = torch.zeros(100,self.num_classes,1,1)
                    targets[:, klass] = 1
                if self.args.cuda:
                    targets = Variable(targets.cuda(), volatile=True) if self.is_C else None
                images = G(self.fixed_noise, targets) if self.is_C else G(self.fixed_noise)
                if not klass in examples.keys():
                    examples[klass] = images
                else:
                    examples[klass] = torch.cat((examples[klass],images), dim=0)

            # Dont save more than the required number of classes
            if save and idx <= self.args.gan_save_classes:
                self.saveResults(examples[klass][0:100], name + "_C" + str(klass), False)
        return examples

    def updateFrozenGenerator(self):
        self.G.eval()
        self.fixed_G = copy.deepcopy(self.G)
        for param in self.fixed_G.parameters():
            param.requires_grad = False

    def unfreezeFrozenGenerator(self):
        self.G.train()
        for param in self.G.parameters():
            param.requires_grad = True

    def saveResults(self, images, name, is_tensor=False, axis_size=10):
        '''
        Saves the images in a grid of axis_size * axis_size
        '''
        axis_size = int(axis_size)
        _, sub = plt.subplots(axis_size, axis_size, figsize=(5, 5))
        for i, j in itertools.product(range(axis_size), range(axis_size)):
            sub[i, j].get_xaxis().set_visible(False)
            sub[i, j].get_yaxis().set_visible(False)

        for k in range(axis_size * axis_size):
            i = k // axis_size
            j = k % axis_size
            sub[i, j].cla()
            if self.args.dataset == "CIFAR100" or self.args.dataset == "CIFAR10":
                if is_tensor:
                    sub[i, j].imshow((images[k].cpu().numpy().transpose(1, 2, 0) + 1)/2)
                else:
                    sub[i, j].imshow((images[k].cpu().data.numpy().transpose(1, 2, 0) + 1)/2)
            elif self.args.dataset == "MNIST":
                if is_tensor:
                    sub[i, j].imshow(images[k, 0].cpu().numpy(), cmap='gray')
                else:
                    sub[i, j].imshow(images[k, 0].cpu().data.numpy(), cmap='gray')

        plt.savefig(self.experiment.path + "results/" + name + ".png")
        plt.cla()
        plt.clf()
        plt.close()

    def saveGANLosses(self, G_Loss, D_Loss, name='GAN_LOSS'):
        x = range(len(G_Loss))
        plt.plot(x, G_Loss, label='G_loss')
        plt.plot(x, D_Loss, label='D_loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.xlim((0, self.args.gan_epochs[self.increment]))

        plt.savefig(self.experiment.path + name + "_" + str(self.increment) + ".png")
        plt.cla()
        plt.clf()
        plt.close()

    def saveCheckpoint(self, epoch):
        '''
        Saves Generator
        '''
        if epoch == 0:
            return
        print("[*] Saving Generator checkpoint")
        path = self.experiment.path + "checkpoints/"
        torch.save(self.G.state_dict(),
                   '{0}G_inc_{1}_e_{2}.pth'.format(path, self.increment, epoch))

    def loadCheckpoint(self, increment):
        '''
        Loads the latest generator for given increment
        '''
        max_e = -1
        filename = None
        for f in os.listdir(self.args.load_g_ckpt):
            vals = f.split('_')
            incr = int(vals[2])
            epoch = int(vals[4].split('.')[0])
            if incr == increment and epoch > max_e:
                max_e = epoch
                filename = f
        if max_e == -1:
            print('[*] Failed to load checkpoint')
            return False
        path = os.path.join(self.args.load_g_ckpt, filename)
        self.G.load_state_dict(torch.load(path))
        print('[*] Loaded Generator from %s' % path)
        return True

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
                    print("Changing GAN Discriminator learning rate from",
                          currentLr_D, "to", currentLr_D * self.args.gan_gammas[temp])
