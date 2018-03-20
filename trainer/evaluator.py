import numpy as np
import torch
from torch.autograd import Variable
from torchnet.meter import confusionmeter
import torch.nn.functional as F

class EvaluatorFactory():
    def __init__(self):
        pass
    @staticmethod
    def get_evaluator(testType="nmc", cuda=True):
        if testType == "nmc":
            return NearestMeanEvaluator(cuda)
        if testType == "trainedClassifier":
            return softmax_evaluator(cuda)


class NearestMeanEvaluator():
    def __init__(self, cuda):
        self.cuda = cuda
        self.means = None
        self.totalFeatures = np.zeros((100, 1))

    def evaluate(self, model, loader):
        model.eval()
        if self.means is None:
            self.means = np.zeros((100, model.featureSize))
        correct = 0

        for data, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                self.means = self.means.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data, True).unsqueeze(1)
            result = (output.data - self.means.float())
            result = torch.norm(result, 2, 2)
            _, pred = torch.min(result, 1)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        return 100. * correct / len(loader.dataset)

    def getConfusionMatrix(self, model, loader, size):
        model.eval()
        test_loss = 0
        correct = 0
        cMatrix = confusionmeter.ConfusionMeter(size, True)

        for data, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                self.means = self.means.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data, True).unsqueeze(1)
            result = (output.data - self.means.float())
            result = torch.norm(result, 2, 2)
            _, pred = torch.min(result, 1)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            cMatrix.add(pred, target.data.view_as(pred))

        test_loss /= len(loader.dataset)
        img = cMatrix.value()
        return img


    def update_means(self, model, train_loader, classes=100):
        # Set the mean to zero
        if self.means is None:
            self.means = np.zeros((100, model.featureSize))
        self.means *= 0
        self.classes = classes
        # Remove the magic number 64
        self.means = np.zeros((classes, model.featureSize))
        self.totalFeatures = np.zeros((classes, 1)) + 1
        print("Computing means")
        # Iterate over all train Dataset
        for batch_id, (data, target) in enumerate(train_loader):
            # Get features for a minibactch
            if self.cuda:
                data = data.cuda()
            features = model.forward(Variable(data), True)
            # Convert result to a numpy array
            featuresNp = features.data.cpu().numpy()
            # Accumulate the results in the means array
            # print (self.means.shape,featuresNp.shape)
            np.add.at(self.means, target, featuresNp)
            # Keep track of how many instances of a class have been seen. This should be an array with all elements = classSize
            np.add.at(self.totalFeatures, target, 1)

        # Divide the means array with total number of instaces to get the average
        self.means = self.means / self.totalFeatures
        self.means = torch.from_numpy(self.means).unsqueeze(0)
        self.means =  self.means / torch.norm(self.means, 2, 1).unsqueeze(1)
        print("Mean vectors computed")
        # Return
        return


class softmax_evaluator():
    def __init__(self, cuda):
        self.cuda = cuda
        self.means = None
        self.totalFeatures = np.zeros((100, 1))

    def evaluate(self, model, loader):
        model.eval()
        correct = 0

        for data, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        return 100. * correct / len(loader.dataset)

    def getConfusionMatrix(self, model, loader, size):
        model.eval()
        test_loss = 0
        correct = 0
        cMatrix = confusionmeter.ConfusionMeter(size, True)

        for data, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            cMatrix.add(pred, target.data.view_as(pred))

        test_loss /= len(loader.dataset)
        img = cMatrix.value()
        return img