import torch
import torch.optim as optim
import torch.utils.data as td
import trainer.classifierTrainer as t
import utils.utils as ut
import trainer.classifierFactory as tF

class trainer():
    def __init__(self, args, dataset, classifierTrainer, model, trainIterator,
                 testIterator, trainDatasetLoader, experiment):
        self.args = args
        self.dataset = dataset
        self.classifierTrainer = classifierTrainer
        self.model = model
        self.trainIterator = trainIterator
        self.testIterator = testIterator
        self.trainDatasetLoader = trainDatasetLoader
        self.experiment = experiment

    def train(self):
        x = []
        y = []
        y1 = []
        trainY = []
        leftOver = []
        myTestFactory = tF.classifierFactory()
        nmc = myTestFactory.getTester("nmc", self.args.cuda)

        if not self.args.sortby == "none":
            print("Sorting by", self.args.sortby)
            self.trainDatasetLoader.sortByImportance(self.args.sortby)

        for classGroup in range(0, self.dataset.classes, self.args.step_size):
            self.classifierTrainer.setupTraining()
            self.classifierTrainer.incrementClasses(classGroup)

            epoch = 0
            for epoch in range(0, self.args.epochs_class):
                self.classifierTrainer.updateLR(epoch)
                self.classifierTrainer.train()
                if epoch % self.args.log_interval == 0:
                    print("Train Classifier",
                          self.classifierTrainer.evaluate(self.trainIterator))
                    print("Test Classifier",
                          self.classifierTrainer.evaluate(self.testIterator))
            self.classifierTrainer.updateFrozenModel()

            nmc.updateMeans(self.model, self.trainIterator, self.args.cuda,
                            self.dataset.classes)

            tempTrain = nmc.classify(self.model, self.trainIterator,
                                     self.args.cuda, True)
            trainY.append(tempTrain)

            # Saving confusion matrix
            ut.saveConfusionMatrix(int(classGroup / self.args.step_size) *
                                   self.args.epochs_class + epoch,
                                   self.experiment.path + "CONFUSION",
                                   self.model, self.args, self.dataset,
                                   self.testIterator)

            # Computing test error for graphing
            testY = nmc.classify(self.model, self.testIterator,
                                 self.args.cuda, True)
            y.append(testY)

            print("Train NMC", tempTrain)
            print("Test NMC", testY)

            y1.append(self.classifierTrainer.evaluate(self.testIterator))
            x.append(classGroup + self.args.step_size)

            ut.plotAccuracy(self.experiment, x,
                            [("NMC", y), ("Trained Classifier", y1)],
                            self.dataset.classes + 1, self.args.name)
