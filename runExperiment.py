from __future__ import print_function

import argparse

import torch
import torch.utils.data as td

import dataHandler
import experiment as ex
import model
import plotter as plt
import trainer

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[45, 60, 68],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='disable distillation loss')
parser.add_argument('--distill-only-exemplars', action='store_true', default=False,
                    help='Only compute the distillation loss on images from the examplar set')
parser.add_argument('--no-random', action='store_true', default=False,
                    help='Disable random shuffling of classes')
parser.add_argument('--no-herding', action='store_true', default=False,
                    help='Disable herding for NMC')
parser.add_argument('--seeds', type=int, nargs='+', default=[200],
                    help='Seeds values to be used')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet32",
                    help='model type to be used. Example : resnet32, resnet20, densenet, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--outputDir', default="../",
                    help='Directory to store the results; the new folder will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--no-upsampling', action='store_true', default=True,
                    help='Do not do upsampling.')
parser.add_argument('--alpha', type=float, default=0.2, help='Weight given to new classes vs old classes in loss')
parser.add_argument('--decay', type=float, default=0.00004, help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--T', type=int, default=3, help='Tempreture used for softening the targets')
parser.add_argument('--memory-budgets', type=int,  nargs='+', default=[2000],
                    help='How many images can we store at max. 0 will result in fine-tuning')
parser.add_argument('--epochs-class', type=int, default=70, help='Number of epochs for each increment')
parser.add_argument('--dataset', default="CIFAR100", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument('--lwf', action='store_true', default=False,
                    help='Use learning without forgetting. Ignores memory-budget '
                         '("Learning with Forgetting," Zhizhong Li, Derek Hoiem)')
parser.add_argument('--rand', action='store_true', default=False,
                    help='Replace exemplars with random instances')
parser.add_argument('--adversarial', action='store_true', default=False,
                    help='Replace exemplars with adversarial instances')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


dataset = dataHandler.DatasetFactory.get_dataset(args.dataset)


# Checks to make sure parameters are sane
if args.step_size<2:
    print("Step size of 1 will result in no learning;")
    assert False

for seed in args.seeds:
    for m in args.memory_budgets:
        args.memory_budget = m

        if args.lwf:
            args.memory_budget = 0


        args.seed = seed
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        # Loader used for training data
        train_dataset_loader = dataHandler.IncrementalLoader(dataset.train_data.train_data, dataset.train_data.train_labels,
                                                             dataset.labels_per_class_train,
                                                             dataset.classes, [], transform=dataset.train_transform,
                                                             cuda=args.cuda, oversampling=not args.no_upsampling,
                                                             )
        # Special loader use to compute ideal NMC; i.e, NMC that using all the data points to compute the mean embedding
        train_dataset_loader_nmc = dataHandler.IncrementalLoader(dataset.train_data.train_data,
                                                             dataset.train_data.train_labels,
                                                             dataset.labels_per_class_train,
                                                             dataset.classes, [], transform=dataset.train_transform,
                                                             cuda=args.cuda, oversampling=not args.no_upsampling,
                                                             )
        # Loader for test data.
        test_dataset_loader = dataHandler.IncrementalLoader(dataset.test_data.test_data, dataset.test_data.test_labels,
                                                            dataset.labels_per_class_test, dataset.classes,
                                                            [], transform=dataset.test_transform, cuda=args.cuda,
                                                            )

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        # Iterator to iterate over training data.
        train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                                     batch_size=args.batch_size, shuffle=True, **kwargs)
        # Iterator to iterate over all training data (Equivalent to memory-budget = infitie
        train_iterator_nmc = torch.utils.data.DataLoader(train_dataset_loader_nmc,
                                                     batch_size=args.batch_size, shuffle=True, **kwargs)
        # Iterator to iterate over test data
        test_iterator = torch.utils.data.DataLoader(
            test_dataset_loader,
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # Get the required model
        myModel = model.ModelFactory.get_model(args.model_type, args.dataset)
        if args.cuda:
            myModel.cuda()

        # Define an experiment.
        my_experiment = ex.experiment(args.name, args)

        # Define the optimizer used in the experiment
        optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.decay, nesterov=True)

        # Trainer object used for training
        my_trainer = trainer.Trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer, train_iterator_nmc)


        # Remove this parameters somehow.
        x = []
        y = []
        y1 = []
        train_y = []

        nmc_ideal_cum = []

        nmc = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)
        nmc_ideal = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)

        t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier", args.cuda)

        # Loop that incrementally adds more and more classes
        for class_group in range(0, dataset.classes, args.step_size):
            print ("SEED:",seed, "MEMORY_BUDGET:", m, "CLASS_GROUP:", class_group)
            my_trainer.setup_training()
            # Add new classes to the train, train_nmc, and test iterator
            my_trainer.increment_classes(class_group)
            my_trainer.update_frozen_model()
            epoch = 0
            import progressbar

            # Running epochs_class epochs
            for epoch in range(0, args.epochs_class):
                my_trainer.update_lr(epoch)
                my_trainer.train(epoch)
                if epoch % args.log_interval == (args.log_interval-1):
                    tError = t_classifier.evaluate(myModel, train_iterator)
                    print ("Current Epoch:", epoch)
                    print("Train Classifier:", tError)
                    print("Test Classifier:", t_classifier.evaluate(myModel, test_iterator))

            # Evaluate the learned classifier
            img = None

            adv = trainer.DisguisedFoolingSampleGeneration(my_trainer.model, 0.8, args.cuda, train_iterator)
            img = adv.generate()



            y1.append(t_classifier.evaluate(myModel, test_iterator))

            # Update means using the train iterator; this is iCaRL case
            nmc.update_means(myModel, train_iterator, dataset.classes)
            # Update mean using all the data. This is equivalent to memory_budget = infinity
            nmc_ideal.update_means(myModel, train_iterator_nmc, dataset.classes)
            # Compute the the nmc based classification results
            tempTrain = nmc.evaluate(myModel, train_iterator)
            train_y.append(tempTrain)

            testY = nmc.evaluate(myModel, test_iterator)
            testY_ideal = nmc_ideal.evaluate(myModel, test_iterator)
            y.append(testY)
            nmc_ideal_cum.append(testY_ideal)

            # Compute confusion matrices of all three cases (Learned classifier, iCaRL, and ideal NMC)
            tcMatrix = t_classifier.get_confusion_matrix(myModel, test_iterator, dataset.classes)
            nmcMatrix = nmc.get_confusion_matrix(myModel, test_iterator, dataset.classes)
            nmcMatrixIdeal = nmc_ideal.get_confusion_matrix(myModel, test_iterator, dataset.classes)

            # Printing results
            print("Train NMC", tempTrain)
            print("Test NMC", testY)

            # Store the resutls in the my_experiment object; this object should contain all the information required to reproduce the results.
            x.append(class_group + args.step_size)

            my_experiment.results["NMC"] = [x, y]
            my_experiment.results["Trained Classifier"] = [x, y1]
            my_experiment.results["Train Error Classifier"] = [x, train_y]
            my_experiment.results["Ideal NMC"] = [x, nmc_ideal_cum]
            my_experiment.store_json()

            # Finally, plotting the results;
            my_plotter = plt.Plotter()

            #
            my_plotter.saveImage(img, my_experiment.path + "GENERATEDImg",
                                 int(class_group / args.step_size) * args.epochs_class + epoch)
            my_plotter.saveImage(oimg, my_experiment.path + "GENERATEDImgOrig",
                                 int(class_group / args.step_size) * args.epochs_class + epoch)
            # Plotting the confusion matrices
            my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,my_experiment.path+"tcMatrix", tcMatrix)
            my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch, my_experiment.path+"nmcMatrix",
                                  nmcMatrix)
            my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                  my_experiment.path + "nmcMatrixIdeal",
                                  nmcMatrixIdeal)

            # Plotting the line diagrams of all the possible cases
            my_plotter.plot(x, y, title=args.name, legend="NMC")
            my_plotter.plot(x, nmc_ideal_cum, title=args.name, legend="Ideal NMC")
            my_plotter.plot(x, y1, title=args.name, legend="Trained Classifier")
            my_plotter.plot(x, train_y, title=args.name, legend="Trained Classifier Train Set")

            # Saving the line plot
            my_plotter.save_fig(my_experiment.path, dataset.classes + 1)
