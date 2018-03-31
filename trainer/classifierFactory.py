import numpy as np
import torch
from torch.autograd import Variable
from torchnet.meter import confusionmeter


class classifierFactory():
    def __init__(self):
        pass

    def getTester(self, testType="nmc", cuda=True):
        if testType == "nmc":
            return NearestMeanClassifier(cuda)


class NearestMeanClassifier():
    def __init__(self, cuda):
        self.cuda = cuda
        self.means = np.zeros((100, 64))
        self.totalFeatures = np.zeros((100, 1))

    def classify(self, model, test_loader, cuda, verbose=False):
        model.eval()
        test_loss = 0
        correct = 0
        cMatrix = confusionmeter.ConfusionMeter(self.classes, True)

        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
                self.means = self.means.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data, True).unsqueeze(1)
            result = (output.data - self.means.float())
            result = torch.norm(result, 2, 2)

            _, pred = torch.min(result, 1)

            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            cMatrix.add(pred, target.data.view_as(pred))
            # 0/0
        test_loss /= len(test_loader.dataset)

        return 100. * correct / len(test_loader.dataset)

    def updateMeans(self, model, train_loader, cuda, classes=100, old_classes=None, is_C=False):
        # Set the mean to zero
        model.eval()
        self.means *= 0
        self.classes = classes
        self.means = np.zeros((classes, 64))
        self.totalFeatures = np.zeros((classes, 1)) + 1
        print("Computing means")
        if (is_C == False):
            print("NMC means on non-conditional GAN not supported")
        # Iterate over all train dataset
        for batch_id, (data, target) in enumerate(train_loader):
            # Get features for a minibactch
            if cuda:
                data = data.cuda()
            features = model.forward(Variable(data), True)

            if (old_classes != None and is_C == False):
                #old_targets = torch.zeros(target.shape[0]).byte()
                #for klass in old_classes:
                #    old_targets += (target == klass)
                #new_targets = (old_targets == 0)
            #TODO get features for all the examples
            # targets = targets*new_targets + predictions*old_targets
            # Do targets[:, new_targets] = predictions[:, old_targets] instead of all above...

            # Convert result to a numpy array
            featuresNp = features.data.cpu().numpy()
            # Accumulate the results in the means array
            np.add.at(self.means, target, featuresNp)
            # Keep track of how many instances of a class have been seen. This should be an array with all elements = classSize
            np.add.at(self.totalFeatures, target, 1)

        # Divide the means array with total number of instaces to get the average
        self.means = self.means / self.totalFeatures

        self.means = torch.from_numpy(self.means).unsqueeze(0)
        print("Mean vectors computed")
        # Return
        return
