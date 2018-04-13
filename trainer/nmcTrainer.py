import torch
import torch.optim as optim
import torch.utils.data as td
import trainer.classifierTrainer as t
import utils.utils as ut
import trainer.classifierFactory as tF

class Trainer():
    def __init__(self, args, dataset, classifier_trainer, model, train_iterator,
                 test_iterator, train_dataset_loader, experiment):
        self.args = args
        self.dataset = dataset
        self.classifier_trainer = classifier_trainer
        self.model = model
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.train_dataset_loader = train_dataset_loader
        self.experiment = experiment

    def train(self):
        x = []
        y = []
        y1 = []
        train_y = []
        left_over = []
        my_test_factory = tF.ClassifierFactory()
        nmc = my_test_factory.get_tester("nmc", self.args.cuda)

        if not self.args.sortby == "none":
            print("Sorting by", self.args.sortby)
            self.train_dataset_loader.sort_by_importance(self.args.sortby)

        for class_group in range(0, self.dataset.classes, self.args.step_size):
            self.classifier_trainer.setup_training()
            self.classifier_trainer.increment_classes(class_group)

            epoch = 0
            for epoch in range(0, self.args.epochs_class):
                self.classifier_trainer.update_lr(epoch)
                self.classifier_trainer.train()
                if epoch % self.args.log_interval == 0:
                    print("Train Classifier",
                          self.classifier_trainer.evaluate(self.train_iterator))
                    print("Test Classifier",
                          self.classifier_trainer.evaluate(self.test_iterator))
            self.classifier_trainer.update_frozen_model()

            nmc.update_means(self.model, self.train_iterator, self.args.cuda,
                            self.dataset.classes)

            temp_train = nmc.classify(self.model, self.train_iterator,
                                     self.args.cuda, True)
            train_y.append(temp_train)

            # Saving confusion matrix
            ut.saveConfusionMatrix(int(class_group / self.args.step_size) *
                                   self.args.epochs_class + epoch,
                                   self.experiment.path + "CONFUSION",
                                   self.model, self.args, self.dataset,
                                   self.test_iterator)

            # Computing test error for graphing
            test_y = nmc.classify(self.model, self.test_iterator,
                                 self.args.cuda, True)
            y.append(test_y)

            print("Train NMC", temp_train)
            print("Test NMC", test_y)

            y1.append(self.classifier_trainer.evaluate(self.test_iterator))
            x.append(class_group + self.args.step_size)

            ut.plotAccuracy(self.experiment, x,
                            [("NMC", y), ("Trained Classifier", y1)],
                            self.dataset.classes + 1, self.args.name)
