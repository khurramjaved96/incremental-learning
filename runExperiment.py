from __future__ import print_function

import argparse

import torch
import torch.optim as optim
import torch.utils.data as td

import dataHandler.datasetFactory as dF
import dataHandler.incrementalLoader as dL
import experiment.experiment as ex
import model.modelFactory as mF
import trainer.classifierTrainer as t
import trainer.nmcTrainer as nt
import trainer.ganTrainer as gt

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[49, 63],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='disable distillation loss')
parser.add_argument('--no-random', action='store_true', default=False,
                    help='Disable random shuffling of classes')
parser.add_argument('--no-herding', action='store_true', default=False,
                    help='Disable herding for NMC')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet32",
                    help='model type to be used. Example : resnet32, resnet20, densenet, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--sortby', default="none",
                    help='Examplars sorting strategy')
parser.add_argument('--decay', type=float, default=0.00004, help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=2, help='How many classes to add in each increment')
parser.add_argument('--memory-budget', type=int, default=2000, help='How many images can we store at max')
parser.add_argument('--epochs-class', type=int, default=70, help='Number of epochs for each increment')
parser.add_argument('--dataset', default="CIFAR100", help='dataset to be used; example CIFAR, MNIST')
parser.add_argument('--no-upsampling', action='store_true', default=False,
                    help='Do not do upsampling.')
parser.add_argument('--process', default="nmc", help='Process to be used to prevent forgetting; Example: nmc, cdcgan, dcgan, wgan')

parser.add_argument('--gan-epochs', type=int, nargs='+', default=[50, 30, 20, 20, 20], help='Epochs for each increment for training the GANs')
parser.add_argument('--gan-d', type=int, default=64, help='GAN feature size multiplier')
parser.add_argument('--gan-lr', type=float, default=0.0002, help='Learning Rate for training the GANs')
parser.add_argument('--gan-batch-size', type=int, default=128, help='Batch Size for training the GANs')
parser.add_argument('--gan-num-examples', type=int, default=1000, help='Number examples GAN will generate for each class')
parser.add_argument('--gan-schedule', type=int, nargs='+', default=[11, 16],
                    help='Decrease GAN learning rate at these epochs.')
parser.add_argument('--gan-gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--persist-gan', action='store_true', default=False,
                    help='GAN is not thrown away and trained from scratch each increment')
parser.add_argument('--gan-img-save-interval', type=int, default=5, help='Save generator samples every x epochs')
parser.add_argument('--d-iter', type=int, default=1, help='Number of iterations of discriminatori/critic for each iteration of generator.')
parser.add_argument('--ckpt-interval', type=int, default=1000, help='After how many epochs should the Generator be saved')
parser.add_argument('--load-g-ckpt', default="", help='Path to folder which contains Generator ckpts')
parser.add_argument('--T', type=int, default=1, help='Temperature for Distillation')
parser.add_argument('--save-g-ckpt', default=False, action='store_true', help='Whether the Generator ckpt will be saved or not')
parser.add_argument('--gan-save-classes', default=10, type=int, help='Number of classes whose images will be saved every gan-img-save-interval')
parser.add_argument('--label-smoothing', default=False, action='store_true', help='Whether to use one sided label smoothing in GAN')
parser.add_argument('--minibatch-discrimination', default=False, action='store_true', help='Whether to use minibatch discrimination layer')
parser.add_argument('--ideal-nmc', default=False, action='store_true', help='Whether to calculate ideal nmc')
parser.add_argument('--optimize-features', default=False, action='store_true', help='Whether to minimize the distance between generated and real embeddings')
parser.add_argument('--optimize-feat-epochs', type=int, default=20, help='How many epochs to run optimize-features for')
parser.add_argument('--optimize-feat-lr', type=float, default=0.00001, help='lr for optimize-features')
parser.add_argument('--joint-gan-obj', default=False, action='store_true', help='Whether to jointly train GAN and minimize the logit distance')
parser.add_argument('--joint-gan-alpha', type=float, default=1, help='Contribution of logit distance minimizer in GAN loss')
parser.add_argument('--ac-distill', default=False, action='store_true', help='Whether to use ACGAN\'s discriminator outputs in distillation')
parser.add_argument('--filter-using-disc', default=False, action='store_true', help='Whether to use discriminator to filter generated samples')
parser.add_argument('--filter-val', type=float, default=0.8, help='Value to be used when filtering using discriminator')

args = parser.parse_args()

if args.process == "gan" and args.dataset == "MNIST" and len(args.gan_epochs) < 10//args.step_size:
    print("ERROR: Number of values in gan-epochs must be greater than number of increments")
    assert False

if args.process == "wgan" and args.d_iter == 1:
    print("NOTICE: Recommended to set --d_iter to 5 for WGAN")

if not args.minibatch_discrimination:
    print("NOTICE: Not using minibatch discrimination, is this intended?")

print("Remember: Set decay 10x smaller on MNIST, was performing better")

args.cuda = not args.no_cuda and torch.cuda.is_available()

dataset = dF.DatasetFactory.get_dataset(args.dataset)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


train_dataset_loader = dL.IncrementalLoader(args.dataset, dataset.train_data.train_data,
                                          dataset.train_data.train_labels,
                                          dataset.labels_per_class_train,
                                          dataset.classes, [], transform=dataset.train_transform,
                                          cuda=args.cuda, oversampling=args.no_upsampling,
                                          alt_transform=dataset.alt_transform
                                          )

test_dataset_loader = dL.IncrementalLoader(args.dataset, dataset.test_data.test_data,
                                         dataset.test_data.test_labels,
                                         dataset.labels_per_class_test, dataset.classes,
                                         [], transform=dataset.test_transform, cuda=args.cuda,
                                         oversampling=args.no_upsampling
                                         )

train_dataset_loader_ideal = None
if args.ideal_nmc:
    train_dataset_loader_ideal = dL.IncrementalLoader(args.dataset, dataset.train_data.train_data,
                                                   dataset.train_data.train_labels,
                                                   dataset.labels_per_class_train,
                                                   dataset.classes, [], transform=dataset.train_transform,
                                                   cuda=args.cuda, oversampling=args.no_upsampling
                                                   )


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                            batch_size=args.batch_size, shuffle=True, **kwargs)
test_iterator = torch.utils.data.DataLoader(test_dataset_loader,
                                           batch_size=args.test_batch_size, shuffle=True, **kwargs)

train_iterator_ideal = None
if args.ideal_nmc:
    train_iterator_ideal = torch.utils.data.DataLoader(train_dataset_loader_ideal,
                                                     batch_size=args.batch_size, shuffle=True, **kwargs)

my_factory = mF.ModelFactory()
model = my_factory.get_model(args.model_type, args.dataset)
if args.cuda:
    model.cuda()

my_experiment = ex.Experiment(args.name, args)

optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                      weight_decay=args.decay, nesterov=True)

classifier_trainer = t.Trainer(train_iterator, test_iterator, dataset, model,
                              args, optimizer, train_iterator_ideal)

if args.process == "nmc":
    trainer = nt.Trainer(args, dataset, classifier_trainer, model,
                         train_iterator, test_iterator, train_dataset_loader,
                         my_experiment)

else:
    trainer = gt.Trainer(args, dataset, classifier_trainer, model,
                         train_iterator, test_iterator, train_dataset_loader,
                         my_factory, my_experiment, train_iterator_ideal, train_dataset_loader_ideal)

trainer.train()
