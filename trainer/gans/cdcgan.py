import time
import itertools

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from trainer.gans.gan import GAN
from torch.autograd import Variable
import trainer.gans.gutils as gutils

class CDCGAN(GAN):
    '''
    CDCGAN Trainer
    __init__ in the base class
    '''

    def train(self, G, D, active_classes, increment):
        '''
        G: Generator
        D: Discriminator
        active_classes: All classes trained on so far
        increment: Which increment number are we on (0 indexed)
        K: Total number of classes in the dataset (just for convenience)
        '''
        g_losses = []
        d_losses = []
        print("ACTIVE CLASSES: ", active_classes)

        criterion = nn.BCELoss()
        g_opt = optim.Adam(G.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(D.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))

        #Matrix of shape [K,K,1,1] with 1s at positions
        #where shape[0]==shape[1]
        K = self.total_classes #We can use len(active_classes) in case of no-persist
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
        # Count G and D iters
        a = 0
        b = 0
        print("Starting GAN Training")
        #-------------Start Epoch-----------#
        for epoch in range(int(self.args.gan_epochs[increment])):
            G.train()
            d_losses_e = []
            g_losses_e = []
            dist_losses_e = []
            start_time = time.time()
            gutils.update_lr(epoch, g_opt, d_opt,
                             self.args.gan_schedule,
                             self.args.gan_gammas)

            #Iterate over examples that the classifier Trainer just iterated on
            for batch_idx, (image, label) in enumerate(self.train_iterator):
                #Handle smaller last batch size
                batch_size = image.shape[0]
                #Save one real sample each epoch
                if not one_sample_saved:
                    gutils.save_results(self.args, image, "sample_E" + str(epoch), True,
                                        np.sqrt(self.args.batch_size),
                                        self.experiment)
                    one_sample_saved = True

                #Make vectors of ones and zeros of same shape as output by
                #Discriminator so that it can be used in BCELoss
                smoothing_val = 0
                if self.args.label_smoothing:
                    smoothing_val = 0.1
                d_like_real = torch.ones(batch_size) - smoothing_val
                d_like_fake = torch.zeros(batch_size)
                if self.args.cuda:
                    d_like_real = Variable(d_like_real.cuda())
                    d_like_fake = Variable(d_like_fake.cuda())

                #-------------Train Discriminator-----------#
                ##Train using real images
                D.zero_grad()
                a = a + 1
                #Shape [batch_size, 10, 32, 32]. Each entry at d_labels[0]
                #contains 32x32 matrix of 1s inside d_labels[label] index
                #and 32x32 matrix of 0s otherwise
                d_labels = d_vec[label]
                if self.args.cuda:
                    image = Variable(image.cuda())
                    d_labels = Variable(d_labels.cuda())
                #Discriminator output for real image and labels
                d_output_real = D(image, d_labels).squeeze()

                ##Train using fake images
                g_random_noise = torch.randn((batch_size, 100))
                g_random_noise = g_random_noise.view(-1, 100, 1, 1)
                #Generating random batch_size of labels from those present in activeClass
                random_labels = torch.from_numpy(np.random.choice(active_classes, batch_size))
                #Convert labels to appropriate shapes
                g_random_labels = g_vec[random_labels]
                d_random_labels = d_vec[random_labels]
                if self.args.cuda:
                    g_random_noise  = Variable(g_random_noise.cuda())
                    g_random_labels = Variable(g_random_labels.cuda())
                    d_random_labels = Variable(d_random_labels.cuda())
                #Generating fake images and passing them to discriminator
                g_output = G(g_random_noise, g_random_labels)
                #Detach gradient from Generator
                g_output = g_output.detach()
                d_output_fake = D(g_output, d_random_labels).squeeze()

                #Calculate BCE loss
                d_real_loss = criterion(d_output_real, d_like_real)
                d_fake_loss = criterion(d_output_fake, d_like_fake)
                d_loss = d_real_loss + d_fake_loss

                #Perform a backward step
                d_losses_e.append(d_loss_total.cpu().data.numpy())
                d_loss.backward()
                d_opt.step()

                #-------------Train Generator-----------#
                #TODO Disabled regenerating of noise, check if it still works
                b = b + 1
                G.zero_grad()
                g_output = G(g_random_noise, g_random_labels)
                d_output = D(g_output, d_random_labels).squeeze()

                #Calculate BCE loss
                g_loss = criterion(d_output, d_like_real)
                total_loss = g_loss

                #Jointly optimizes the GAN Generator loss and lowers
                #the euclidean distance between features of generated
                #and real images
                if self.args.joint_gan_obj:
                    model = self.fixed_classifier
                    euclidean_dist = nn.PairwiseDistance(2)
                    # Generate features for real and fake images
                    output_fake = model.forward(g_output, True)
                    output_real = model.forward(image, True)
                    # Calculate euclidean distance
                    distance_loss = torch.mean(euclidean_dist(output_fake, output_real))
                    total_loss = g_loss + (self.args.joint_gan_alpha * distance_loss)
                    dist_losses_e.append(distance_loss.cpu().data.numpy())

                #Backward step
                total_loss.backward()
                g_losses_e.append(g_loss.cpu().data.numpy())
                g_opt.step()

            #-------------End Epoch-----------#
            #Print Stats and save results
            mean_g = (sum(g_losses_e)/len(g_losses_e))
            mean_d = (sum(d_losses_e)/len(d_losses_e))
            mean_dist = None
            if self.args.joint_gan_obj:
                mean_dist = (sum(dist_losses_e)/len(dist_losses_e))
            g_losses.append(mean_g)
            d_losses.append(mean_d)

            if epoch % self.args.gan_img_save_interval == 0:
                #Generate some examples for visualizing
                gutils.generate_examples(self.args, G, 100,
                                         active_classes,
                                         self.total_classes,
                                         self.fixed_noise,
                                         self.experiment,
                                         "Inc" + str(increment) + "_E" + str(epoch),
                                         True, True)
                #Plot GAN losses
                gutils.save_gan_losses(g_losses, d_losses,
                                       self.args.gan_epochs[increment],
                                       increment,
                                       self.experiment)

            #Save ckpt if interval is satisfied
            if self.args.save_g_ckpt and epoch % self.args.ckpt_interval == 0:
                gutils.save_checkpoint(epoch, increment, self.experiment, G)

            print("[GAN] Epoch:", epoch,
                  "G_iters:", b,
                  "D_iters:", a,
                  "g_loss:", mean_g,
                  "d_loss:", mean_d,
                  "dist_Loss:", mean_dist,
                  "Time taken:", time.time() - start_time)
