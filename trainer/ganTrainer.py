import torch
import utils.utils as ut
import torch.optim as optim
import torch.utils.data as td
import model.modelFactory as mF
import trainer.classifierTrainer as t
import trainer.classifierFactory as tF

class trainer():
    def __init__(self, args, dataset, classifierTrainer, model, G, D,
                trainIterator, testIterator, trainDatasetLoader, experiment):
        self.args = args
        self.dataset = dataset
        self.classifierTrainer = classifierTrainer
        self.model = model
        self.G = G
        self.D = D
        self.trainIterator = trainIterator
        self.testIterator = testIterator
        self.trainDatasetLoader = trainDatasetLoader
        self.experiment = experiment

    def train(self):
        x = []
        y = []

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
            #self.classifierTrainer.updateFrozenModel()

            # Saving confusion matrix
            ut.saveConfusionMatrix(int(classGroup / self.args.step_size) *
                                   self.args.epochs_class + epoch,
                                   self.experiment.path + "CONFUSION",
                                   self.model, self.args, self.dataset,
                                   self.testIterator)

            y.append(self.classifierTrainer.evaluate(self.testIterator))
            x.append(classGroup + self.args.step_size)

            ut.plotAccuracy(self.experiment, x,
                            [("Trained Classifier",y)],
                            self.dataset.classes + 1, self.args.name)

    def trainGan(self):
        activeClasses = trainIterator.dataset.activeClasses
