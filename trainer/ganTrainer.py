import torch
import pickle
import itertools
import utils.utils as ut
import torch.optim as optim
import torch.utils.data as td
import trainer.classifierTrainer as t
import trainer.classifierFactory as tF

class trainer():
    def __init__(self, args, dataset, classifierTrainer, model, trainIterator,
                 testIterator, trainDatasetLoader, modelFactory, experiment):
        self.args = args
        self.dataset = dataset
        self.classifierTrainer = classifierTrainer
        self.model = model
        self.trainIterator = trainIterator
        self.testIterator = testIterator
        self.trainDatasetLoader = trainDatasetLoader
        self.modelFactory = modelFactory
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
            #self.classifierTrainer.updateFrozenModel() Do we need this?

            #Get a new Generator and Discriminator
            #TODO What if we kept the Discriminator?
            G, D = self.modelFactory.getModel("cdcgan", args.dataset)
            if args.cuda:
                G.cuda()
                D.cuda()

            self.trainGan()

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
        activeClasses = self.trainIterator.dataset.activeClasses

        criterion = nn.BCELoss()
        G_Opt = optim.Adam(G.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        D_Opt = optim.Adam(D.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        #Matrix of shape [10,10,1,1] with 1s at positions
        #where shape[0]==shape[1]
        GVec = torch.zeros(10, 10)
        GVec = GVec.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

        #Matrix of shape [10,10,28,28] with 28x28 matrix of 1s
        #where shape[0]==shape[1]
        DVec = torch.zeros([10, 10, 32, 32])
        for i in range(10):
            DVec[i, i, :, :] = 1

        print("Starting GAN Training")
        for epoch in range(int(args.gan_epochs)):
            self.updateLR(epoch, G_Opt, D_Opt)

    #TODO Merge this with updateLR function in classifierTrainer
    #TODO Add the new args to runExperiment
    def updateLR(self, epoch, G_Opt, D_Opt):
        for temp in range(0, len(self.args.gan_schedule)):
            if self.args.gan_schedule[temp] == epoch:
                #Update Generator LR
                for param_group in G_Opt.param_groups:
                    currentLr_G = param_group['lr']
                    param_group['lr'] = currentLr_G * self.args.gan_gammas[temp]
                    print("Changing GAN Generator learning rate from",
                          currentLr_G, "to", currentLr_G * self.args.gan_gammas[temp])
                #Update Discriminator LR
                for param_group in D_Opt.param_groups:
                    currentLr_D = param_group['lr']
                    param_group['lr'] = currentLr_D * self.args.gan_gammas[temp]
                    print("Changing GAN Generator learning rate from",
                          currentLr_D, "to", currentLr_D * self.args.gan_gammas[temp])





