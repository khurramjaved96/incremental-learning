from __future__ import print_function

import argparse

import torch
import torch.optim as optim
import torch.utils.data as td

import dataHandler.datasetFactory as dF
import dataHandler.incrementalLoader as dL
import experiment.experiment as ex
import model.modelFactory as mF
import plotter.plotter as plt
import trainer.classifierFactory as tF
import trainer.classifierTrainer as t
import trainer.nmcTrainer as nt
import trainer.ganTrainer as gt
import utils.utils as ut

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 30, 40, 50, 57],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2, 0.2, 0.2],
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
parser.add_argument('--decay', type=float, default=0.00001, help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--memory-budget', type=int, default=2000, help='How many images can we store at max')
parser.add_argument('--epochs-class', type=int, default=60, help='Number of epochs for each increment')
parser.add_argument('--dataset', default="CIFAR100", help='dataset to be used; example CIFAR, MNIST')
parser.add_argument('--process', default="nmc", help='Process to be used to prevent forgetting; Example: nmc, gan')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = dF.datasetFactory.getDataset(args.dataset)

trainDatasetLoader = dL.incrementalLoader(dataset.trainData.train_data, dataset.trainData.train_labels,
                                          dataset.labelsPerClassTrain,
                                          dataset.classes, [], transform=dataset.trainTransform,
                                          cuda=args.cuda,
                                          )

testDatasetLoader = dL.incrementalLoader(dataset.testData.test_data, dataset.testData.test_labels,
                                         dataset.labelsPerClassTest, dataset.classes,
                                         [], transform=dataset.testTransform, cuda=args.cuda,
                                         )

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

trainIterator = torch.utils.data.DataLoader(trainDatasetLoader,
                                            batch_size=args.batch_size, shuffle=True, **kwargs)
testIterator = torch.utils.data.DataLoader(testDatasetLoader,
                                           batch_size=args.test_batch_size, shuffle=True, **kwargs)

G = D = None
myFactory = mF.modelFactory()
model = myFactory.getModel(args.model_type, args.dataset)

if args.process == "gan":
    G, D = myFactory.getModel("cdcgan", args.dataset)

if args.cuda:
    model.cuda()
    if args.process == "gan":
        G.cuda()
        D.cuda()

myExperiment = ex.experiment(args)

optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                      weight_decay=args.decay, nesterov=True)

classifierTrainer = t.trainer(trainIterator, testIterator, dataset, model, args, optimizer)

if args.process == "nmc":
    trainer = nt.trainer(args, dataset, classifierTrainer, model, trainIterator, testIterator, trainDatasetLoader,  myExperiment)
    trainer.train()

if args.process == "gan":
    trainer = gt.trainer()
