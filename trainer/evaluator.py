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

    def get_confusion_matrix(self, model, loader, size):
        model.eval()
        test_loss = 0
        correct = 0
        # Get the confusion matrix object
        cMatrix = confusionmeter.ConfusionMeter(size, True)

        for data, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                self.means = self.means.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data, True).unsqueeze(1)
            result = (output.data - self.means.float())
            result = torch.norm(result, 2, 2)
            # NMC for classification
            _, pred = torch.min(result, 1)
            # Evaluate results
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # Add the results in appropriate places in the matrix.
            cMatrix.add(pred, target.data.view_as(pred))

        test_loss /= len(loader.dataset)
        # Get 2d numpy matrix to remove the dependency of other code on confusionmeter
        img = cMatrix.value()
        return img


    def update_means(self, model, train_loader, classes=100):
        # Set the mean to zero
        if self.means is None:
            self.means = np.zeros((classes, model.featureSize))
        self.means *= 0
        self.classes = classes
        self.means = np.zeros((classes, model.featureSize))
        self.totalFeatures = np.zeros((classes, 1)) + .001
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

        # Divide the means array with total number of instan    ces to get the average
        # print ("Total instances", self.totalFeatures)
        self.means = self.means / self.totalFeatures
        self.means = torch.from_numpy(self.means)
        # Normalize the mean vector
        self.means = self.means / torch.norm(self.means, 2, 1).unsqueeze(1)
        self.means[self.means != self.means] = 0
        self.means = self.means.unsqueeze(0)

        print("Mean vectors computed")
        # Return
        return


class softmax_evaluator():
    def __init__(self, cuda):
        self.cuda = cuda
        self.means = None
        self.totalFeatures = np.zeros((100, 1))

    def evaluate(self, model, loader, scale=None, thres=False, older_classes=None, step_size=10):
        scale = np.copy(scale)
        model.eval()
        correct = 0
        if scale is not None:
            scale = scale/np.max(scale)
            # print ("Gets here")
            scaleTemp = np.copy(scale)
            if thres:
                for x in range(0, len(scale)):
                    temp = 0
                    for y in range(0, len(scale)):
                        if x == y:
                            pass
                        else:
                            temp=temp+(scale[y]/scale[x])
                        scaleTemp[x] = temp
                scale = scaleTemp
            else:
                scale = 1 / scale
            # scale[len(older_classes)+step_size:len(scale)] = 1
            # scale = np.log(scale)
            # print (scale)
            # scale = scale-1
            scale = scale/np.linalg.norm(scale, 1)
            scale = torch.from_numpy(scale).unsqueeze(0)
            if self.cuda:
                scale = scale.cuda()
        for data, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            if thres:
                output = model(data)
                output = output*Variable(scale.float())
            elif scale is not None:
                # print("Gets here, getting outputs")
                output = model(data, scale = Variable(scale.float()))
            else:
                output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        return 100. * correct / len(loader.dataset)

    def get_confusion_matrix(self, model, loader, size, scale=None, older_classes=None, step_size=10):
        scale = np.copy(scale)
        model.eval()
        test_loss = 0
        correct = 0
        cMatrix = confusionmeter.ConfusionMeter(size, True)

        if scale is not None:
            scale = scale/np.max(scale)
            # print ("Gets here")
            scale = 1 / scale
            # scale[len(older_classes)+step_size:len(scale)] = 1
            # scale = np.log(scale)
            # print (scale)
            # scale = scale-1
            # scale[len(older_classes) + step_size:len(scale)] = 1
            scale = torch.from_numpy(scale).unsqueeze(0)
            if self.cuda:
                scale = scale.cuda()


        for data, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            if scale is not None:
                # print("Gets here, getting outputs")
                output = model(data, scale = Variable(scale.float()))
            else:
                output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            cMatrix.add(pred.squeeze(), target.data.view_as(pred).squeeze())

        test_loss /= len(loader.dataset)
        img = cMatrix.value()
        return img
