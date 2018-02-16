from __future__ import print_function
import argparse
import torch.utils.data as td
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import dataHandler.incrementalLoaderCifar as dL
import model.modelFactory as mF
import copy
import plotter.plotter as plt
import trainer.classifierFactory as tF
import trainer.trainer as t
import utils.utils as ut

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[20,30,40,50,57], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2,0.2,0.2,0.2,0.2], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='argument to enable/disable distillation loss')
parser.add_argument('--no-random', action='store_true', default=False,
                    help='Do not shuffle the classes')
parser.add_argument('--no-herding', action='store_true', default=False,
                    help='To do herding or not to do herding')
parser.add_argument('--no-upsampling', action='store_true', default=False,
                    help='argument to enable/disable upsampling')
parser.add_argument('--oversampling', action='store_true', default=False,
                    help='Should we use oversampling')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type',  default="resnet32",
                    help='model type to be used')
parser.add_argument('--name',  default="noname",
                    help='Name of the experiment')
parser.add_argument('--sortby',  default="Kennard-Stone",
                    help='Name of the experiment')
parser.add_argument('--decay', type=float, default=0.00001 , help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--memory-budget', type=int, default=2000, help='How many images can we store at max')
parser.add_argument('--epochs-class', type=int, default=60, help='Number of epochs for each increment')
parser.add_argument('--classes', type=int, default=100, help='Total classes (after all the increments)')
parser.add_argument('--depth', type=int, default=32, help='depth of the model; only valid for resnet')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

schedule = args.schedule
gammas = args.gammas
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


experimentName = ut.constructExperimentName(args)

# Mean and STD of Cifar-100 dataset.
# To do : Remove the hard-coded mean and just compute it once using the data
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(0.5,0.5,0.5,0.5), transforms.RandomCrop(32, padding=6),torchvision.transforms.RandomRotation((-10,10)), transforms.ToTensor(),
     transforms.Normalize(mean, std)])

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

train_data = datasets.CIFAR100("data", train=True, transform=train_transform, download=True)
test_data = datasets.CIFAR100("data", train=False, transform=test_transform, download=True)


trainDatasetFull = dL.incrementalLoaderCifar(train_data.train_data,train_data.train_labels, 500,100,[],transform=train_transform,cuda= args.cuda,  oversampling=args.oversampling)
testDataset = dL.incrementalLoaderCifar(test_data.test_data,test_data.test_labels, 100,100,[],transform=test_transform, cuda= args.cuda,  oversampling=args.oversampling)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader_full = torch.utils.data.DataLoader(trainDatasetFull,
    batch_size=args.batch_size, shuffle=True, **kwargs)


test_loader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


# Selecting model
myFactory = mF.modelFactory()
model = myFactory.getModel(args.model_type,args.classes)
if args.cuda:
    model.cuda()


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

stepSize = args.step_size
leftOver = []
limitedset=[]
totalExmp = args.memory_budget
epochsPerClass=args.epochs_class
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


for classGroup in range(0, args.classes, stepSize):
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
        #print ("Limiting class", val,"to",int(totalExmp/len(leftOver)))
        if args.no_herding:
            trainDatasetFull.limitClass(val,int(totalExmp/len(leftOver)))
        else:
            print ("Sorting by herding")
            trainDatasetFull.limitClassAndSort(val,int(totalExmp/len(leftOver)),modelFixed)
        limitedset.append(val)

    for temp in range(classGroup, classGroup+stepSize):
        popVal = allClasses.pop()
        trainDatasetFull.addClasses(popVal)
        testDataset.addClasses(popVal)
        leftOver.append(popVal)
    epoch=0
    for epoch in range(0,epochsPerClass):
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
    ut.saveConfusionMatrix(int(classGroup/stepSize)*epochsPerClass + epoch,experimentName+"CONFUSION", model, args, test_loader)
    print ("Test NMC")
    y.append(nmc.classify(model,test_loader,args.cuda, True))
    y1.append(t.test(test_loader, model, args))
    x.append(classGroup+stepSize)

    myPlotter = plt.plotter()
    myPlotter.plot(x,y, title=args.name, legend="NCM")
    myPlotter.plot(x, y1, title=args.name, legend="Trained Classifier")

    myPlotter.saveFig(experimentName+".jpg")
