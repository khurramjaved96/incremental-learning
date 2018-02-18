import torch
import torch.optim as optim
import torch.utils.data as td
import trainer.classifierTrainer as t
import utils.utils as ut
import trainer.classifierFactory as tF

class trainer():
    def train(self, args, dataset, classifierTrainer, model, trainIterator, testIterator, experiment):
        x = []
        y = []
        y1 = []
        trainY = []
        leftOver = []
        myTestFactory = tF.classifierFactory()
        nmc = myTestFactory.getTester("nmc", args.cuda)

        if not args.sortby == "none":
            print("Sorting by", args.sortby)
            trainDatasetLoader.sortByImportance(args.sortby)

        for classGroup in range(0, dataset.classes, args.step_size):

            classifierTrainer.setupTraining()

            classifierTrainer.incrementClasses(classGroup)

            epoch = 0
            for epoch in range(0, args.epochs_class):
                classifierTrainer.updateLR(epoch)
                classifierTrainer.train()
                if epoch % args.log_interval == 0:
                    print("Train Classifier", classifierTrainer.evaluate(trainIterator))
                    print("Test Classifier", classifierTrainer.evaluate(testIterator))
            classifierTrainer.updateFrozenModel()
            nmc.updateMeans(model, trainIterator, args.cuda, dataset.classes)

            tempTrain = nmc.classify(model, trainIterator, args.cuda, True)
            trainY.append(tempTrain)

            # Saving confusion matrix
            ut.saveConfusionMatrix(int(classGroup / args.step_size) * args.epochs_class + epoch,
                                   experiment.path + "CONFUSION", model, args, dataset, testIterator)
            # Computing test error for graphing
            testY = nmc.classify(model, testIterator, args.cuda, True)
            y.append(testY)

            print("Train NMC", tempTrain)
            print("Test NMC", testY)

            y1.append(classifierTrainer.evaluate(testIterator))
            x.append(classGroup + args.step_size)

            ut.plotAccuracy(experiment, x,
                            [("NMC", y), ("Trained Classifier",y1)],
                            dataset.classes + 1, args.name)

