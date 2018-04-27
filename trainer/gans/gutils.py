import os
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def save_results(args, images, name, is_tensor=False, axis_size=10, experiment=None):
    '''
    Saves the images in a grid of axis_size * axis_size
    args: args
    images: Dict of images to save
    is_tensor: Whether the images are a torch tensor
    axis_size: Number of images on each axis
    experiment: Experiment object (needed to get destination path)
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
        if args.dataset == "CIFAR100" or args.dataset == "CIFAR10":
            if is_tensor:
                sub[i, j].imshow((images[k].cpu().numpy().transpose(1, 2, 0) + 1)/2)
            else:
                sub[i, j].imshow((images[k].cpu().data.numpy().transpose(1, 2, 0) + 1)/2)
        elif args.dataset == "MNIST":
            if is_tensor:
                sub[i, j].imshow(images[k, 0].cpu().numpy(), cmap='gray')
            else:
                sub[i, j].imshow(images[k, 0].cpu().data.numpy(), cmap='gray')

    plt.savefig(experiment.path + "results/" + name + ".png")
    plt.cla()
    plt.clf()
    plt.close()


def generate_examples(
        args, G, num_examples, active_classes, total_classes,
        noise, experiment, name="", save=False, is_cond=False, D=None):
    '''
    Returns a dict[class] of generated samples.
    In case of Non-Conditional GAN, the samples in the dict are random, they do
    not correspond to the keys in the dict
    Just passing in random noise to the generator and storing the results in dict
    Generates a batch of 100 examples at a time
    args: args
    num_examples: Total number of examples to generate
    active_classes: List of all classes trained on till now
    total_classes: Total number of classes in the dataset
    noise: A noise vector of size [100,100,1,1] to generate examples
    experiment: Experiment object
    save: If True, also save samples of generated images to disk
    is_cond: If True, use the label information too (Only use with supported GANs)
    '''
    print("Note: Ignoring the fixed noise")
    G.eval()
    if D is not None:
        D.eval()
    examples = {}
    num_iter = 0
    for idx, klass in enumerate(active_classes):
        while ((not klass in examples.keys()) or (len(examples[klass]) < num_examples)):
            num_iter += 1
            if args.process == "cdcgan":
                targets = torch.zeros(100, total_classes, 1, 1)
                targets[:, klass] = 1
                noise = torch.randn(100, 100, 1, 1)
            elif args.process == "acgan":
                targets = np.zeros((100, total_classes))
                targets[:, klass] = 1
                nz_noise = np.random.normal(0, 1, (100, 100))
                combined_noise = np.append(targets, nz_noise, axis=1)
                noise = torch.from_numpy(combined_noise)
                noise = noise.view(100, 100+total_classes, 1, 1).float()
            else:
                noise = torch.randn(100, 100, 1, 1)
            if args.cuda:
                noise = Variable(noise.cuda(), volatile=True)
                if args.process == "cdcgan":
                    targets = Variable(targets.cuda(), volatile=True)
            if args.process == "cdcgan":
                images = G(noise, targets)
            else:
                images = G(noise)
            if args.filter_using_disc and D is not None:
                d_output = D(images)
                indices = (d_output[0] > args.filter_val).nonzero().squeeze()
                if indices.dim() == 0:
                    continue
                images = torch.index_select(images, 0, indices)
            if not klass in examples.keys():
                examples[klass] = images
            else:
                examples[klass] = torch.cat((examples[klass],images), dim=0)

        # Dont save more than the required number of classes
        if save and idx <= args.gan_save_classes and args.gan_num_examples > 0:
            save_results(args, examples[klass][0:100],
                         name + "_C" + str(klass),
                         False, 10, experiment)
        # Trim extra examples
    if D is not None:
        for klass in active_classes:
            examples[klass] = examples[klass][0:num_examples]
        print("Examples matching the filter: ", num_examples / (num_iter * 100), "%")
    return examples


def save_gan_losses(g_loss, d_loss, epochs, increment, experiment, name='GAN_LOSS'):
    '''
    Plots the GAN loss curves for both G and D
    '''
    x = range(len(g_loss))
    plt.plot(x, g_loss, label='G_loss')
    plt.plot(x, d_loss, label='D_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.xlim((0, epochs))

    plt.savefig(experiment.path + name + "_" + str(increment) + ".png")
    plt.cla()
    plt.clf()
    plt.close()


def save_checkpoint(epoch, increment, experiment, G, D=None):
    '''
    Saves the ckpt to disk
    '''
    if epoch == 0:
        return
    print("[*] Saving Generator checkpoint")
    path = experiment.path + "checkpoints/"
    torch.save(G.state_dict(),
               '{0}G_inc_{1}_e_{2}.pth'.format(path, increment, epoch))
    if D is None:
        return
    print("[*] Saving Discriminator checkpoint")
    path = experiment.path + "checkpoints/"
    torch.save(D.state_dict(),
               '{0}D_inc_{1}_e_{2}.pth'.format(path, increment, epoch))


def load_checkpoint(g_ckpt_path, increment, G):
    '''
    Loads the latest generator for given increment
    g_ckpt_path: path to checkpoints folder
    increment: current increment number
    G: Generator
    '''
    max_e = -1
    filename = None
    for f in os.listdir(g_ckpt_path):
        vals = f.split('_')
        #TODO Load the discriminator too
        if vals[0] != "G":
            continue
        incr = int(vals[2])
        epoch = int(vals[4].split('.')[0])
        if incr == increment and epoch > max_e:
            max_e = epoch
            filename = f
    if max_e == -1:
        print('[*] Failed to load checkpoint')
        return False
    path = os.path.join(g_ckpt_path, filename)
    G.load_state_dict(torch.load(path))
    print('[*] Loaded Generator from %s' % path)
    return True


def update_lr(epoch, g_opt, d_opt, gan_schedule, gan_gammas):
    '''
    Update the lr for both optimizers
    '''
    for temp in range(0, len(gan_schedule)):
        if gan_schedule[temp] == epoch:
            #Update Generator LR
            for param_group in g_opt.param_groups:
                current_lr_g = param_group['lr']
                param_group['lr'] = current_lr_g * gan_gammas[temp]
                print("Changing GAN Generator learning rate from",
                      current_lr_g, "to", current_lr_g * gan_gammas[temp])
            #Update Discriminator LR
            for param_group in d_opt.param_groups:
                current_lr_d = param_group['lr']
                param_group['lr'] = current_lr_d * gan_gammas[temp]
                print("Changing GAN Discriminator learning rate from",
                      current_lr_d, "to", current_lr_d * gan_gammas[temp])
