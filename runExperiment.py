from __future__ import print_function

import argparse

import torch
import torch.optim as optim
import torch.utils.data as td

import dataHandler.DatasetFactory as dF
import dataHandler.IncrementalLoader as dL
import experiment.experiment as ex
import model.ModelFactory as mF
import plotter.plotter as plt
import trainer as tF
import trainer.trainer as t
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
parser.add_argument('--outputDir', default="../",
                    help='Directory to store the results; the new folder will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--no-upsampling', action='store_true', default=False,
                    help='Do not do upsampling.')
parser.add_argument('--decay', type=float, default=0.00001, help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--memory-budget', type=int, default=2000,
                    help='How many images can we store at max. 0 will result in fine-tuning')
parser.add_argument('--epochs-class', type=int, default=60, help='Number of epochs for each increment')
parser.add_argument('--dataset', default="CIFAR100", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument('--lwf', action='store_true', default=False,
                    help='Use learning without forgetting. Ignores memory-budget '
                         '("Learning with Forgetting," Zhizhong Li, Derek Hoiem)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.lwf:
    args.memory_budget = 0
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = dF.DatasetFactory.get_dataset(args.dataset)

train_dataset_loader = dL.IncrementalLoader(dataset.train_data.train_data, dataset.train_data.train_labels,
                                            dataset.labels_per_class_train,
                                            dataset.classes, [], transform=dataset.train_transform,
                                            cuda=args.cuda, oversampling=not args.no_upsampling,
                                            )

test_dataset_loader = dL.IncrementalLoader(dataset.test_data.test_data, dataset.test_data.test_labels,
                                           dataset.labels_per_class_test, dataset.classes,
                                           [], transform=dataset.test_transform, cuda=args.cuda,
                                           )

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                             batch_size=args.batch_size, shuffle=True, **kwargs)
test_iterator = torch.utils.data.DataLoader(
    test_dataset_loader,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = mF.ModelFactory.get_model(args.model_type, args.dataset)
if args.cuda:
    model.cuda()

my_experiment = ex.experiment(args.name, args)

optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                      weight_decay=args.decay, nesterov=True)

my_trainer = t.Trainer(train_iterator, test_iterator, dataset, model, args, optimizer)

x = []
y = []
y1 = []
train_y = []

nmc = tF.EvaluatorFactory.get_evaluator("nmc", args.cuda)
t_classifier = tF.EvaluatorFactory.get_evaluator("trainedClassifier", args.cuda)

if not args.sortby == "none":
    print("Sorting by", args.sortby)
    train_dataset_loader.sort_by_importance(args.sortby)

for class_group in range(0, dataset.classes, args.step_size):

    my_trainer.setup_training()

    my_trainer.increment_classes(class_group)
    my_trainer.update_frozen_model()
    epoch = 0
    for epoch in range(0, args.epochs_class):
        my_trainer.update_lr(epoch)
        my_trainer.train()
        if epoch % args.log_interval == 0:
            print("Train Classifier", t_classifier.evaluate(model, train_iterator))
            print("Test Classifier", t_classifier.evaluate(model, test_iterator))

    nmc.update_means(model, train_iterator, dataset.classes)

    tempTrain = nmc.evaluate(model, train_iterator)
    train_y.append(tempTrain)

    # Saving confusion matrix
    ut.save_confusion_matrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                             my_experiment.path + "CONFUSION", model, args, dataset, test_iterator)
    # Computing test error for graphing
    testY = nmc.evaluate(model, test_iterator)
    y.append(testY)

    print("Train NMC", tempTrain)
    print("Test NMC", testY)

    y1.append(t_classifier.evaluate(model, test_iterator))
    x.append(class_group + args.step_size)

    my_experiment.results["NCM"] = [x, y]
    my_experiment.results["Trained Classifier"] = [x, y1]
    my_experiment.results["Train Error Classifier"] = [x, train_y]
    my_experiment.store_json()

    my_plotter = plt.Plotter()
    my_plotter.plot(x, y, title=args.name, legend="NCM")
    my_plotter.plot(x, y1, title=args.name, legend="Trained Classifier")

    my_plotter.save_fig(my_experiment.path + "Overall" + ".jpg", dataset.classes + 1)
