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

class Trainer():
    def __init__(self, args, dataset, classifier_trainer, model, train_iterator,
                 test_iterator, train_loader, model_factory, experiment, train_iterator_ideal, train_loader_ideal):
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

        test_factory = tF.ClassifierFactory()
        nmc = test_factory.get_tester("nmc", self.args.cuda)
        if self.args.ideal_nmc:
            ideal_nmc = test_factory.get_tester("nmc", self.args.cuda)

        for class_group in range(0, self.dataset.classes, self.args.step_size):
            self.classifier_trainer.setup_training()
            self.classifier_trainer.increment_classes(class_group)
            #Get new iterator with reduced batch_size
            if class_group > 0:
                self.increment = self.increment + 1
                self.old_classes = self.classifier_trainer.older_classes
                self.examples = self.generate_examples(self.fixed_g,
                                                      self.args.gan_num_examples,
                                                      self.old_classes,
                                                      "Final-Inc"+str(self.increment-1),
                                                      True)
                #TODO put trainLoader
                self.train_iterator.dataset.replace_data(self.examples,
                                                         self.args.gan_num_examples)
                # Send examples to CPU
                if self.is_cond:
                    for k in self.examples:
                        self.examples[k] = self.examples[k].data.cpu()

            ######################
            # Train Classifier
            ######################
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

            # Using NMC classifier
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

            #####################
            # Train GAN
            ####################
            if self.G == None or not self.args.persist_gan:
                self.G, self.D = self.model_factory.get_model(self.args.process,
                                                            self.args.dataset,
                                                            self.args.minibatch_discrimination,
                                                            self.args.gan_d)
                if self.args.cuda:
                    self.G = self.G.cuda()
                    self.D = self.D.cuda()
            is_loaded = False
            if self.args.load_g_ckpt != '':
                is_loaded = self.load_checkpoint(self.increment)
            if not is_loaded:
                self.train_gan(self.G, self.D, self.is_cond, self.num_classes)
            if self.args.optimize_features:
                self.optimize_features()
            self.update_frozen_generator()
            if self.args.save_g_ckpt:
                self.save_checkpoint(self.args.gan_epochs[self.increment])

            # Saving confusion matrix
            ut.saveConfusionMatrix(int(class_group / self.args.step_size) *
                                   self.args.epochs_class + epoch,
                                   self.experiment.path + "CONFUSION",
                                   self.model, self.args, self.dataset,
                                   self.test_iterator)

            # Plot
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

    def optimize_features(self):
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


    def train_gan(self, G, D, is_cond, K):
        g_losses = []
        d_losses = []
        active_classes = self.train_iterator.dataset.active_classes
        print("ACTIVE CLASSES: ", active_classes)

        # Switch to alternate transformations while training GAN
        self.train_loader.do_alt_transform = True

        #TODO Change batchsize of dataIterator here to gan_batch_size
        if self.args.process == "wgan":
            if self.args.gan_lr > 5e-5 or len(self.args.gan_schedule) > 1:
                print(">>> NOTICE: Did you mean to set GAN lr/schedule to this value?")
            g_opt = optim.RMSprop(G.parameters(), lr=self.args.gan_lr)
            d_opt = optim.RMSprop(D.parameters(), lr=self.args.gan_lr)
        elif self.args.process == "dcgan" or self.args.process == "cdcgan":
            criterion = nn.BCELoss()
            g_opt = optim.Adam(G.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))
            d_opt = optim.Adam(D.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))

        #Matrix of shape [K,K,1,1] with 1s at positions
        #where shape[0]==shape[1]
        if is_cond:
            tensor = []
            g_vec = torch.zeros(K, K)
            for i in range(K):
                tensor.append(i)
            g_vec = g_vec.scatter_(1, torch.LongTensor(tensor).view(K,1),
                                 1).view(K, K, 1, 1)
            #Matrix of shape [K,K,32,32] with 32x32 matrix of 1s
            #where shape[0]==shape[1]
            d_vec = torch.zeros([K, K, 32, 32])
            for i in range(K):
                d_vec[i, i, :, :] = 1

        one_sample_saved = False
        a = 0
        b = 0
        print("Starting GAN Training")
        for epoch in range(int(self.args.gan_epochs[self.increment])):
            #######################
            #Start Epoch
            #######################
            G.train()
            d_losses_e = []
            g_losses_e = []
            dist_losses_e = []
            start_time = time.time()
            self.update_lr(epoch, g_opt, d_opt)

            #Iterate over examples that the classifier Trainer just iterated on
            for batch_idx, (image, label) in enumerate(self.train_iterator):
                batch_size = image.shape[0]
                if not one_sample_saved:
                    self.save_results(image, "sample_E" + str(epoch), True, np.sqrt(self.args.batch_size))
                    one_sample_saved = True

                #Make vectors of ones and zeros of same shape as output by
                #Discriminator so that it can be used in BCELoss
                if self.args.process == "dcgan" or self.args.process == "cdcgan":
                    smoothing_val = 0
                    if self.args.label_smoothing:
                        smoothing_val = 0.1
                    d_like_real = torch.ones(batch_size) - smoothing_val
                    d_like_fake = torch.zeros(batch_size)
                    if self.args.cuda:
                        d_like_real = Variable(d_like_real.cuda())
                        d_like_fake = Variable(d_like_fake.cuda())
                ##################################
                #Train Discriminator
                ##################################
                #Train with real image and labels
                D.zero_grad()
                a = a + 1

                #Shape [batch_size, 10, 32, 32]. Each entry at d_labels[0]
                #contains 32x32 matrix of 1s inside d_labels[label] index
                #and 32x32 matrix of 0s otherwise
                d_labels = d_vec[label] if is_cond else None
                if self.args.cuda:
                    image    = Variable(image.cuda())
                    d_labels = Variable(d_labels.cuda()) if is_cond else None

                #Discriminator output for real image and labels
                d_output_real = D(image, d_labels).squeeze() if is_cond else D(image).squeeze()

                #Train with fake image and labels
                g_random_noise = torch.randn((batch_size, 100))
                g_random_noise = g_random_noise.view(-1, 100, 1, 1)

                if is_cond:
                    #Generating random batch_size of labels from amongst
                    #labels present in activeClass
                    random_labels = torch.from_numpy(np.random.choice(active_classes,
                                                                      batch_size))
                    #Convert labels to appropriate shapes
                    g_random_labels = g_vec[random_labels]
                    d_random_labels = d_vec[random_labels]

                if self.args.cuda:
                    g_random_noise  = Variable(g_random_noise.cuda())
                    g_random_labels = Variable(g_random_labels.cuda()) if is_cond else None
                    d_random_labels = Variable(d_random_labels.cuda()) if is_cond else None

                g_output = G(g_random_noise, g_random_labels) if is_cond else G(g_random_noise)
                g_output = g_output.detach()
                d_output_fake = D(g_output, d_random_labels).squeeze() if is_cond else D(g_output).squeeze()

                if self.args.process == "wgan":
                    d_loss = -(torch.mean(d_output_real) - torch.mean(d_output_fake))

                elif self.args.process == "dcgan" or self.args.process == "cdcgan":
                    d_real_loss = criterion(d_output_real, d_like_real)
                    d_fake_loss = criterion(d_output_fake, d_like_fake)
                    d_loss = d_real_loss + d_fake_loss

                d_losses_e.append(d_loss)
                d_loss.backward()
                d_opt.step()

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
                g_random_noise = torch.randn((batch_size, 100))
                g_random_noise = g_random_noise.view(-1, 100, 1, 1)

                if is_cond:
                    random_labels = torch.from_numpy(np.random.choice(active_classes,
                                                                      batch_size))
                    g_random_labels = g_vec[random_labels]
                    d_random_labels = d_vec[random_labels]

                if self.args.cuda:
                    g_random_noise  = Variable(g_random_noise.cuda())
                    g_random_labels = Variable(g_random_labels.cuda()) if is_cond else None
                    d_random_labels = Variable(d_random_labels.cuda()) if is_cond else None

                g_output = G(g_random_noise, g_random_labels) if is_cond else G(g_random_noise)
                d_output = D(g_output, d_random_labels).squeeze() if is_cond else D(g_output).squeeze()

                if self.args.process == "wgan":
                    g_loss = -torch.mean(d_output)
                elif self.args.process == "dcgan" or self.args.process == "cdcgan":
                    g_loss = criterion(d_output, d_like_real)
                total_loss = g_loss

                if self.args.joint_gan_obj:
                    model = self.classifier_trainer.model_fixed
                    euclidean_dist = nn.PairwiseDistance(2)
                    # Generate features for real and fake images
                    output_fake = model.forward(g_output, True)
                    output_real = model.forward(image, True)
                    # Calculate euclidean distance
                    distance_loss = torch.mean(euclidean_dist(output_fake, output_real))
                    total_loss = g_loss + (self.args.joint_gan_alpha * distance_loss)
                    dist_losses_e.append(distance_loss)

                total_loss.backward()
                g_losses_e.append(g_loss)
                g_opt.step()

            #############################
            #End epoch
            #Print Stats and save results
            #############################
            mean_g = (sum(g_losses_e)/len(g_losses_e)).cpu().data.numpy()[0]
            mean_d = (sum(d_losses_e)/len(d_losses_e)).cpu().data.numpy()[0]
            mean_dist = None
            if self.args.joint_gan_obj:
                mean_dist = (sum(dist_losses_e)/len(dist_losses_e)).cpu().data.numpy()[0]
            g_losses.append(mean_g)
            d_losses.append(mean_d)

            if epoch % self.args.gan_img_save_interval == 0:
                self.generate_examples(G, 100, active_classes,
                                      "Inc"+str(self.increment) +
                                      "_E" + str(epoch), True)
                self.save_gan_losses(g_losses, d_losses)

            if self.args.save_g_ckpt and epoch % self.args.ckpt_interval == 0:
                self.save_checkpoint(epoch)
            print("[GAN] Epoch:", epoch,
                  "G_iters:", b,
                  "D_iters:", a,
                  "g_loss:", mean_g,
                  "d_loss:", mean_d,
                  "dist_Loss:", mean_dist,
                  "Time taken:", time.time() - start_time)
        # Switch to default transformation
        self.train_loader.do_alt_transform = False

    def generate_examples(self, G, num_examples, active_classes, name="", save=False):
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
                if self.is_cond:
                    targets = torch.zeros(100,self.num_classes,1,1)
                    targets[:, klass] = 1
                if self.args.cuda:
                    targets = Variable(targets.cuda(), volatile=True) if self.is_cond else None
                images = G(self.fixed_noise, targets) if self.is_cond else G(self.fixed_noise)
                if not klass in examples.keys():
                    examples[klass] = images
                else:
                    examples[klass] = torch.cat((examples[klass],images), dim=0)

            # Dont save more than the required number of classes
            if save and idx <= self.args.gan_save_classes:
                self.save_results(examples[klass][0:100], name + "_C" + str(klass), False)
        return examples

    def update_frozen_generator(self):
        self.G.eval()
        self.fixed_g = copy.deepcopy(self.G)
        for param in self.fixed_g.parameters():
            param.requires_grad = False

    def unfreeze_frozen_generator(self):
        self.G.train()
        for param in self.G.parameters():
            param.requires_grad = True

    def save_results(self, images, name, is_tensor=False, axis_size=10):
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

    def save_gan_losses(self, g_loss, d_loss, name='GAN_LOSS'):
        x = range(len(g_loss))
        plt.plot(x, g_loss, label='G_loss')
        plt.plot(x, d_loss, label='D_loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.xlim((0, self.args.gan_epochs[self.increment]))

        plt.savefig(self.experiment.path + name + "_" + str(self.increment) + ".png")
        plt.cla()
        plt.clf()
        plt.close()

    def save_checkpoint(self, epoch):
        '''
        Saves Generator
        '''
        if epoch == 0:
            return
        print("[*] Saving Generator checkpoint")
        path = self.experiment.path + "checkpoints/"
        torch.save(self.G.state_dict(),
                   '{0}G_inc_{1}_e_{2}.pth'.format(path, self.increment, epoch))

    def load_checkpoint(self, increment):
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

    def update_lr(self, epoch, g_opt, d_opt):
        for temp in range(0, len(self.args.gan_schedule)):
            if self.args.gan_schedule[temp] == epoch:
                #Update Generator LR
                for param_group in g_opt.param_groups:
                    current_lr_g = param_group['lr']
                    param_group['lr'] = current_lr_g * self.args.gan_gammas[temp]
                    print("Changing GAN Generator learning rate from",
                          current_lr_g, "to", current_lr_g * self.args.gan_gammas[temp])
                #Update Discriminator LR
                for param_group in d_opt.param_groups:
                    current_lr_d = param_group['lr']
                    param_group['lr'] = current_lr_d * self.args.gan_gammas[temp]
                    print("Changing GAN Discriminator learning rate from",
                          current_lr_d, "to", current_lr_d * self.args.gan_gammas[temp])
