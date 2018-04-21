import time
import itertools

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from trainer.gans.gan import GAN
from torch.autograd import Variable
import trainer.gans.gutils as gutils

class WGAN(GAN):
    '''
    WGAN Trainer
    __init__ in the base class
    '''

    def train(self, G, D, active_classes, increment):
        '''
        G: Generator
        D: Discriminator
        active_classes: All classes trained on so far
        increment: Which increment number are we on (0 indexed)
        '''
        g_losses = []
        d_losses = []
        print("ACTIVE CLASSES: ", active_classes)

        #Author of WGAN used RMSprop
        if self.args.gan_lr > 5e-5 or len(self.args.gan_schedule) > 1:
            print(">>> NOTICE: Did you mean to set GAN lr/schedule to this value?")
        g_opt = optim.RMSprop(G.parameters(), lr=self.args.gan_lr)
        d_opt = optim.RMSprop(D.parameters(), lr=self.args.gan_lr)

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
                batch_size = image.shape[0]
                if not one_sample_saved:
                    gutils.save_results(self.args, image, "sample_E" + str(epoch), True,
                                        np.sqrt(self.args.batch_size),
                                        self.experiment)
                    one_sample_saved = True

                #-------------Train Discriminator-----------#
                ##Train using real images
                D.zero_grad()
                a = a + 1

                #Discriminator output for real images
                if self.args.cuda:
                    image = Variable(image.cuda())
                d_output_real = D(image).squeeze()

                ##Train using fake images
                g_random_noise = torch.randn((batch_size, 100))
                g_random_noise = g_random_noise.view(-1, 100, 1, 1)
                if self.args.cuda:
                    g_random_noise  = Variable(g_random_noise.cuda())

                #Generating fake images and passing them to discriminator
                g_output = G(g_random_noise)
                #Detach gradient from Generator
                g_output = g_output.detach()
                d_output_fake = D(g_output).squeeze()

                #Calculate WGAN Loss (no BCE here)
                d_loss = -(torch.mean(d_output_real) - torch.mean(d_output_fake))


                #Perform a backward step
                d_losses_e.append(d_loss_total.cpu().data.numpy())
                d_loss.backward()
                d_opt.step()

                #Clamp the parameters (part of WGAN)
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)

                #-------------Train Generator-----------#
                #Train critic (discriminator) more in case of WGAN because the
                #critic needs to be trained to optimality
                if batch_idx % self.args.d_iter != 0:
                    continue

                b = b + 1
                G.zero_grad()
                g_output = G(g_random_noise)
                d_output = D(g_output).squeeze()

                #Calculate WGAN Generator loss
                g_loss = -torch.mean(d_output)
                total_loss = g_loss

                #Jointly optimizes the GAN Generator loss and lowers
                #the euclidean distance between features of generated
                #and real images (default: off)
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
                                         True, False)
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
