import time
import itertools

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from trainer.gans.gan import GAN
from torch.autograd import Variable
from utils import utils
import trainer.gans.gutils as gutils

class ACGAN(GAN):
    '''
    ACGAN Trainer
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
        print("ACTIVE CLASSES: ", active_classes)

        #Define variables
        iters = 0
        a_losses = []
        g_losses = []
        d_losses = []
        one_sample_saved = False
        d_criterion = nn.BCELoss()
        a_criterion = nn.NLLLoss()
        nz = self.total_classes + 100
        nc = 1 if self.args.dataset == "MNIST" else 3
        real_label = 0.9 if self.args.label_smoothing else 1
        fake_label = 0

        #Define tensors, d_label=discriminator label a_label=auxilary label
        inp = torch.FloatTensor(self.args.batch_size, nc, 32, 32)
        noise = torch.FloatTensor(self.args.batch_size, nz, 1, 1)
        d_label = torch.FloatTensor(self.args.batch_size)
        a_label = torch.LongTensor(self.args.batch_size)

        #To cuda and Variable
        if self.args.cuda:
            d_criterion.cuda()
            a_criterion.cuda()
            inp, noise= inp.cuda(), noise.cuda()
            d_label, a_label = d_label.cuda(), a_label.cuda()
        inp, noise = Variable(inp), Variable(noise)
        d_label, a_label = Variable(d_label), Variable(a_label)

        #what is a better way to not forget setting good hyperparams?
        assert self.args.gan_lr == 0.0002
        g_opt = optim.Adam(G.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(D.parameters(), lr=self.args.gan_lr, betas=(0.5, 0.999))

        print("Starting GAN Training")
        #-------------Start Epoch-----------#
        for epoch in range(int(self.args.gan_epochs[increment])):
            acc_e = []
            a_losses_e = []
            d_losses_e = []
            g_losses_e = []
            fake_prob_e = []
            real_prob_e = []
            G.train()
            gutils.update_lr(epoch, g_opt, d_opt, self.args.gan_schedule,
                             self.args.gan_gammas)
            start_time = time.time()
            #Iterate over examples that the classifier Trainer just iterated on
            for batch_idx, (image, label) in enumerate(self.train_iterator):
                iters = iters + 1
                batch_size = image.shape[0]
                if self.args.cuda:
                    image = image.cuda()
                    label = label.cuda()
                if inp.shape[0] != batch_size:
                    inp.data.resize_as_(image)
                    noise.data.resize_(batch_size, nz, 1, 1)
                    d_label.data.resize_(batch_size)
                    a_label.data.resize_(batch_size)
                if not one_sample_saved:
                    one_sample_saved = True
                    gutils.save_results(self.args, image, "sample_E" + str(epoch), True,
                                        np.sqrt(self.args.batch_size), self.experiment)

                #-------------Train Discriminator-----------#
                # https://discuss.pytorch.org/t/how-to-use-the-backward-functions-for-multiple-losses/1826/6 Huh?
                ##Train using real images
                D.zero_grad()
                inp.data.copy_(image)
                d_label.data.fill_(real_label)
                a_label.data.copy_(label)

                d_output, a_output = D(inp)
                d_loss_real = d_criterion(d_output, d_label)
                a_loss_real = a_criterion(a_output, a_label)
                d_real_total = d_loss_real + a_loss_real
                d_real_total.backward()
                real_prob_e.append(d_output.data.mean())
                acc_e.append(utils.compute_acc(a_output, a_label))

                ##Train using fake images
                #Generate random noise
                noise.data.normal_(0, 1)
                #Noise in ACGAN consists of label info + noise
                nz_labels = np.random.choice(active_classes, batch_size)
                nz_noise = np.random.normal(0, 1, (batch_size, 100))
                hot_labels = np.zeros((batch_size, self.total_classes))
                hot_labels[np.arange(batch_size), nz_labels] = 1
                #Combine the two vectors
                combined_noise = np.append(hot_labels, nz_noise, axis=1)
                combined_noise = torch.from_numpy(combined_noise)
                #Insert into the Variables
                noise.data.copy_(combined_noise.view(batch_size, nz, 1, 1))
                d_label.data.fill_(fake_label)
                a_label.data.copy_(torch.from_numpy(nz_labels))

                g_output = G(noise)
                g_output_temp = g_output.detach()
                d_output, a_output = D(g_output_temp)
                d_loss_fake = d_criterion(d_output, d_label)
                a_loss_fake = a_criterion(a_output, a_label)
                d_fake_total = d_loss_fake + a_loss_fake
                d_fake_total.backward()
                d_loss_total = d_real_total + d_fake_total
                fake_prob_e.append(d_output.data.mean())
                d_opt.step()
                d_losses_e.append(d_loss_total.cpu().data.numpy())

                #-------------Train Generator-----------#
                G.zero_grad()
                d_label.data.fill_(real_label)
                d_output, a_output = D(g_output)
                d_loss_g = d_criterion(d_output, d_label)
                a_loss_g = a_criterion(a_output, a_label)
                g_loss_total = d_loss_g + a_loss_g
                g_loss_total.backward()
                g_opt.step()
                g_losses_e.append(g_loss_total.cpu().data.numpy())
            #-------------End Epoch-----------#
            #Print Stats and save results
            mean_g = (sum(g_losses_e)/len(g_losses_e))
            mean_d = (sum(d_losses_e)/len(d_losses_e))
            mean_acc = (sum(acc_e)/len(acc_e))
            mean_prob_real = (sum(real_prob_e)/len(real_prob_e))
            mean_prob_fake = (sum(fake_prob_e)/len(fake_prob_e))
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
                gutils.save_checkpoint(epoch, increment, self.experiment, G, D)
            time_taken = time.time() - start_time
            print("[GAN] Epoch: %d" % epoch,
                  "Iters: %d" % iters,
                  "g_loss: %f" % mean_g,
                  "d_loss: %f" % mean_d,
                  "acc: %f" % mean_acc,
                  "D(x): %f" % mean_prob_real,
                  "D(G(z)): %f" % mean_prob_fake,
                  "Time taken: %f" % time_taken)
