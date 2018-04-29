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
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 35)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[15, 23, 28],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--random-init', action='store_true', default=True,
                    help='To initialize model using previous weights or random weights in each iteration')
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='disable distillation loss')
parser.add_argument('--distill-only-exemplars', action='store_true', default=False,
                    help='Only compute the distillation loss on images from the examplar set')
parser.add_argument('--no-random', action='store_true', default=False,
                    help='Disable random shuffling of classes')
parser.add_argument('--no-herding', action='store_true', default=True,
                    help='Disable herding for NMC')
parser.add_argument('--seeds', type=int, nargs='+', default=[23423],
                    help='Seeds values to be used')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet32",
                    help='model type to be used. Example : resnet32, resnet20, densenet, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--outputDir', default="../",
                    help='Directory to store the results; the new folder will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--upsampling', action='store_true', default=False,
                    help='Do not do upsampling.')
parser.add_argument('--pp', action='store_true', default=False,
                    help='Privacy perserving')

parser.add_argument('--hs', action='store_true', default=False,
                    help='Hierarchical Softmax')

parser.add_argument('--unstructured-size', type=int, default=0, help='Number of epochs for each increment')
parser.add_argument('--no-nl', action='store_true', default=False,
                    help='No Normal Loss')

parser.add_argument('--alphas', type=float, nargs='+', default=[1.0],
                    help='Weight given to new classes vs old classes in loss')
parser.add_argument('--decay', type=float, default=0.00005, help='Weight decay (L2 penalty).')
parser.add_argument('--alpha-increment', type=float, default=1.0, help='Weight decay (L2 penalty).')
parser.add_argument('--l1', type=float, default=0.0, help='Weight decay (L1 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--T', type=float, default=1, help='Tempreture used for softening the targets')
parser.add_argument('--memory-budgets', type=int, nargs='+', default=[40000],
                    help='How many images can we store at max. 0 will result in fine-tuning')
parser.add_argument('--epochs-class', type=int, default=30, help='Number of epochs for each increment')
parser.add_argument('--dataset', default="MNIST", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument('--lwf', action='store_true', default=False,
                    help='Use learning without forgetting. Ignores memory-budget '
                         '("Learning with Forgetting," Zhizhong Li, Derek Hoiem)')
parser.add_argument('--rand', action='store_true', default=False,
                    help='Replace exemplars with random instances')
parser.add_argument('--adversarial', action='store_true', default=False,
                    help='Replace exemplars with adversarial instances')

import logging

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dataset = dataHandler.DatasetFactory.get_dataset(args.dataset)

# Checks to make sure parameters are sane
if args.step_size < 2:
    logging.error("Step size of 1 will result in no learning;")
    assert False

for seed in args.seeds:
    for at in args.alphas:
        args.alpha = at
        for m in args.memory_budgets:
            args.memory_budget = m

            if args.lwf:
                args.memory_budget = 0

            args.seed = seed
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)

            # Loader used for training data
            train_dataset_loader = dataHandler.IncrementalLoader(dataset.train_data.train_data,
                                                                 dataset.train_data.train_labels,
                                                                 dataset.labels_per_class_train,
                                                                 dataset.classes, [], transform=dataset.train_transform,
                                                                 cuda=args.cuda, oversampling=not args.upsampling,
                                                                 )
            # Special loader use to compute ideal NMC; i.e, NMC that using all the data points to compute the mean embedding
            train_dataset_loader_nmc = dataHandler.IncrementalLoader(dataset.train_data.train_data,
                                                                     dataset.train_data.train_labels,
                                                                     dataset.labels_per_class_train,
                                                                     dataset.classes, [],
                                                                     transform=dataset.train_transform,
                                                                     cuda=args.cuda, oversampling=not args.upsampling,
                                                                     )
            # Loader for test data.
            test_dataset_loader = dataHandler.IncrementalLoader(dataset.test_data.test_data,
                                                                dataset.test_data.test_labels,
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

            logger = logging.getLogger('iCARL')
            logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(my_experiment.path + ".log")
            fh.setLevel(logging.DEBUG)

            fh2 = logging.FileHandler("../temp.log")
            fh2.setLevel(logging.DEBUG)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            fh2.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(fh2)
            logger.addHandler(ch)

            # Define the optimizer used in the experiment
            optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                                        weight_decay=args.decay, nesterov=True)

            # Trainer object used for training
            my_trainer = trainer.Trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer,
                                         train_iterator_nmc)

            # Remove this parameters somehow.
            x = []
            y = []
            y1 = []
            train_y = []
            y_scaled = []
            y_gscaled = []
            nmc_ideal_cum = []

            nmc_ideal = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)

            t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier", args.cuda)

            # Loop that incrementally adds more and more classes
            class_group = 0
            my_trainer.increment_classes(class_group)
            my_trainer.update_frozen_model()
            epoch = 0

            # Running epochs_class epochs
            for epoch in range(0, args.epochs_class):
                my_trainer.update_lr(epoch)
                my_trainer.train(epoch)
                # print(my_trainer.threshold)
                if epoch % args.log_interval == (args.log_interval - 1):
                    tError = t_classifier.evaluate(my_trainer.model, train_iterator)
                    print("********CURRENT EPOCH*********", epoch)
                    print("Train Classifier:", tError)
                    print("Test Classifier:", t_classifier.evaluate(my_trainer.model, test_iterator))
                    print("Test Classifier Scaled:",
                          t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.threshold, False,
                                                my_trainer.older_classes, args.step_size))

            testError = t_classifier.evaluate(my_trainer.model, test_iterator)
            testErrorScaled = t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.threshold, False,
                                                    my_trainer.older_classes, args.step_size)
            testErrorGScaled = t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.threshold2, False,
                                                     my_trainer.older_classes, args.step_size)
            y_gscaled.append(testErrorGScaled)

            x.append(0)
            y1.append(testError)
            y_scaled.append(testErrorScaled)
            logging.info("Orig Model Test Error %0.2f", testError)
            logging.info("Orig Model Test Scaled Error %0.2f", testErrorScaled)
            logging.info("Orig Model Test GScaled Error %0.2f", testErrorGScaled)
            my_trainer.update_frozen_model()
            nmc_ideal.update_means(my_trainer.model, train_iterator_nmc, dataset.classes)
            testY_ideal = nmc_ideal.evaluate(my_trainer.model, test_iterator)

            # my_trainer.setup_training()
            nmc_ideal_cum.append(testY_ideal)
            listOfElem = range(10)
            import random
            random.seed(args.seed)
            random.shuffle(listOfElem)
            logging.info("List order" + ",".join([str(i) for i in listOfElem]))
            for xTemp in listOfElem:
                logging.info("Removing class %d", xTemp)
                my_trainer.resetThresh()
                my_trainer.limit_class(xTemp, 0, False)
                my_trainer.randomInitModel()

                for epoch in range(0, args.epochs_class):
                    my_trainer.update_lr(epoch)
                    my_trainer.train(epoch)
                    # print(my_trainer.threshold)
                    if epoch % args.log_interval == (args.log_interval - 1):
                        tError = t_classifier.evaluate(my_trainer.model, train_iterator)
                        logger.info("********CURRENT EPOCH********* %0.2f", epoch)
                        logger.info("Train Classifier: %0.2f", tError)
                        logger.info("Test Classifier: %0.2f", t_classifier.evaluate(my_trainer.model, test_iterator))
                        logger.info("Test Classifier Scaled: %0.2f",
                                    t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.threshold, False,
                                                          my_trainer.older_classes, args.step_size))

                # Evaluate the learned classifier
                img = None

                logger.info("Test Classifier Final: %0.2f", t_classifier.evaluate(my_trainer.model, test_iterator))
                logger.info("Test Classifier Final Scaled: %0.2f",
                            t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.threshold, False,
                                                  my_trainer.older_classes, args.step_size))

                y_scaled.append(t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.threshold, False,
                                                      my_trainer.older_classes, args.step_size))
                y1.append(t_classifier.evaluate(my_trainer.model, test_iterator))

                testErrorGScaled = t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.threshold2, False,
                                                         my_trainer.older_classes, args.step_size)
                logger.info("Test Classifier Final GScaled: %0.2f", testErrorGScaled)

                y_gscaled.append(testErrorGScaled)

                # Update means using the train iterator; this is iCaRL case
                # nmc.update_means(my_trainer.model, train_iterator, dataset.classes)
                # Update mean using all the data. This is equivalent to memory_budget = infinity

                # Compute the the nmc based classification results
                tempTrain = t_classifier.evaluate(my_trainer.model, train_iterator)
                train_y.append(tempTrain)

                # testY1 = nmc.evaluate(my_trainer.model, test_iterator, step_size=args.step_size,  kMean = True)
                # testY = nmc.evaluate(my_trainer.model, test_iterator)
                nmc_ideal.update_means(my_trainer.model, train_iterator_nmc, dataset.classes)
                testY_ideal = nmc_ideal.evaluate(my_trainer.model, test_iterator)

                nmc_ideal_cum.append(testY_ideal)
                # y.append(testY)
                # Compute confusion matrices of all three cases (Learned classifier, iCaRL, and ideal NMC)
                tcMatrix = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)
                tcMatrix_scaled = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes,
                                                                    my_trainer.threshold, my_trainer.older_classes,
                                                                    args.step_size)
                # nmcMatrix = nmc.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)
                nmcMatrixIdeal = nmc_ideal.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)

                # Printing results
                print("Train Claissifier", tempTrain)
                # print("Test NMC", testY)

                # my_trainer.setup_training()

                # Store the resutls in the my_experiment object; this object should contain all the information required to reproduce the results.
                x.append(xTemp + 1)

                my_experiment.results["NMC"] = [x, y]
                my_experiment.results["Trained Classifier"] = [x, y1]
                my_experiment.results["Trained Classifier Scaled"] = [x, y_scaled]
                my_experiment.results["Train Error Classifier"] = [x, train_y]
                my_experiment.results["Ideal NMC"] = [x, nmc_ideal_cum]
                my_experiment.store_json()

                # Finally, plotting the results;
                my_plotter = plt.Plotter()

                # Plotting the confusion matrices
                my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                      my_experiment.path + "tcMatrix" + str(xTemp), tcMatrix)
                my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                      my_experiment.path + "tcMatrix_scaled" + str(xTemp), tcMatrix_scaled)
                my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                      my_experiment.path + "nmcMatrixIdeal" + str(xTemp),
                                      nmcMatrixIdeal)

                # Plotting the line diagrams of all the possible cases
                # my_plotter.plot(x, y, title=args.name, legend="NMC")
                my_plotter.plot(x, y_scaled, title=args.name, legend="Trained Classifier Scaled")
                my_plotter.plot(x, nmc_ideal_cum, title=args.name, legend="Ideal NMC")
                my_plotter.plot(x, y1, title=args.name, legend="Trained Classifier")
                # my_plotter.plot(x, train_y, title=args.name, legend="Trained Classifier Train Set")

                # Saving the line plot
                my_plotter.save_fig(my_experiment.path, dataset.classes + 1)
