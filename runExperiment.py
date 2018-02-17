from __future__ import print_function
import argparse
import torch.utils.data as td
import torch
import torch.optim as optim

import model.modelFactory as mF
import copy
import plotter.plotter as plt
import trainer.classifierFactory as tF
import trainer.trainer as t
import utils.utils as ut
import experiment.experiment as ex
import dataHandler.datasetFactory as dF

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[20,30,40,50,57], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2,0.2,0.2,0.2,0.2], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

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
parser.add_argument('--oversampling', action='store_true', default=False,
                    help='Do oversampling to train unbiased classifier')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type',  default="resnet32",
                    help='model type to be used. Example : resnet32, resnet20, densenet, test')
parser.add_argument('--name',  default="noname",
                    help='Name of the experiment')
parser.add_argument('--sortby',  default="none",
                    help='Examplars sorting strategy')
parser.add_argument('--decay', type=float, default=0.00001 , help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--memory-budget', type=int, default=2000, help='How many images can we store at max')
parser.add_argument('--epochs-class', type=int, default=60, help='Number of epochs for each increment')
parser.add_argument('--classes', type=int, default=100, help='Total classes (after all the increments)')
parser.add_argument('--depth', type=int, default=32, help='depth of the model; only valid for resnet')
parser.add_argument('--dataset', default="MNIST", help='dataset to be used; example CIFAR, MNIST')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

schedule = args.schedule
gammas = args.gammas
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



train_data, trainDatasetFull = dF.datasetFactory.getDataset(args.dataset, args, True)
test_data, testDataset = dF.datasetFactory.getDataset(args.dataset, args, False)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader_full = torch.utils.data.DataLoader(trainDatasetFull,
    batch_size=args.batch_size, shuffle=True, **kwargs)


test_loader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Selecting model
myFactory = mF.modelFactory()
model = myFactory.getModel(args.model_type,args.classes,"MNIST")
if args.cuda:
    model.cuda()

myExperiment = ex.experiment(vars(args))
myExperiment.constructExperimentName(args)

modelFixed = None

optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                weight_decay=args.decay, nesterov=True)
currentLr = args.lr

allClasses = list(range(args.classes))
allClasses.sort(reverse=True)

# Will be important when computing confidence intervals.
import random
if not args.no_random:
    print ("Randomly shuffling classes")
    random.shuffle(allClasses)

leftOver = []
limitedset=[]
distillLoss = False

x = []
y = []
y1 = []
trainY= []
myTestFactory= tF.classifierFactory()
nmc = myTestFactory.getTester("nmc", args.cuda)

if not args.sortby == "none":
    print ("Sorting by", args.sortby)
    trainDatasetFull.sortByImportance(args.sortby)

overallEpoch = 0


for classGroup in range(0, args.classes, args.step_size):
    if classGroup ==0:
        distillLoss=False
    else:
        distillLoss=True
        modelFixed = copy.deepcopy(model)
        for param in modelFixed.parameters():
            param.requires_grad = False
        # model.classifier = nn.Linear(64, 100).cuda()
    for param_group in optimizer.param_groups:
        print ("Setting LR to", args.lr)
        param_group['lr'] = args.lr
        currentLr = args.lr 
    for val in leftOver:

        if args.no_herding:
            trainDatasetFull.limitClass(val,int(args.memory_budget/len(leftOver)))
        else:
            print ("Sorting by herding")
            trainDatasetFull.limitClassAndSort(val,int(args.memory_budget/len(leftOver)),modelFixed)
        limitedset.append(val)

    for temp in range(classGroup, classGroup+args.step_size):
        popVal = allClasses.pop()
        trainDatasetFull.addClasses(popVal)
        testDataset.addClasses(popVal)
        print ("Train Classes", trainDatasetFull.activeClasses)
        leftOver.append(popVal)
    epoch=0
    for epoch in range(0,args.epochs_class):
        overallEpoch+=1
        for temp in range(0, len(schedule)):
            if schedule[temp]==epoch:
                for param_group in optimizer.param_groups:
                    currentLr = param_group['lr']
                    param_group['lr'] = currentLr*gammas[temp]
                    print("Changing learning rate from", currentLr, "to", currentLr*gammas[temp])
                    currentLr*= gammas[temp]
        t.train(optimizer, train_loader_full,limitedset, model, modelFixed, args)
        if epoch%5==0:
            print("Train Classifier", t.test(train_loader_full, model, args))
            print ("Test Classifier", t.test(test_loader, model, args))
    nmc.updateMeans(model, train_loader_full, args.cuda, args.classes)

    tempTrain = nmc.classify(model,train_loader_full,args.cuda, True)
    trainY.append(tempTrain)
    print("Train NMC", tempTrain)
    ut.saveConfusionMatrix(int(classGroup/args.step_size)*args.epochs_class + epoch,myExperiment.path+"CONFUSION", model, args, test_loader)
    testY = nmc.classify(model, test_loader, args.cuda, True)
    y.append(testY)
    print ("Test NMC", testY)

    y1.append(t.test(test_loader, model, args))
    x.append(classGroup+args.step_size)

    myExperiment.results["NCM"] = [x,y]
    myExperiment.results["Trained Classifier"] = [x,y1]

    with open(myExperiment.path+"result", "wb") as f:
        import pickle
        pickle.dump(myExperiment, f)

    myPlotter = plt.plotter()
    myPlotter.plot(x,y, title=args.name, legend="NCM")
    myPlotter.plot(x, y1, title=args.name, legend="Trained Classifier")

    myPlotter.saveFig(myExperiment.path+"Overall"+".jpg", args.classes+1)

