from torchnet.meter import confusionmeter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch

class classifierFactory():
    def __init__(self):
        pass
    def getTester(self, testType="nmc", cuda=True):
        if testType=="nmc":
            return NearestMeanClassifier(cuda)



class NearestMeanClassifier():
    def __init__(self, cuda):
        self.cuda = cuda
        self.means=np.zeros((100, 64))+1e5
        self.totalFeatures = np.zeros((100,1))
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
            print (result)
            _, pred = torch.min(result, 1)
            print (pred)
            print (target)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            cMatrix.add(pred, target.data.view_as(pred))

        test_loss /= len(test_loader.dataset)
        print ("Correct pred", correct, "Total Pre", len(test_loader.dataset))
        if verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        return 100. * correct / len(test_loader.dataset)

    def updateMeans(self, model, train_loader,cuda, classes=100):
        #Set the mean to zero
        self.means*=0
        self.classes = classes
        self.means = np.zeros((classes, 64)) + 1e5
        self.totalFeatures = np.zeros((classes, 1))
        print ("Computing means")
        #Iterate over all train dataset
        for batch_id, (data, target) in enumerate(train_loader):
            #Get features for a minibactch
            if cuda:
                data = data.cuda()
            features = model.forward(Variable(data), True)
            #Convert result into a numpy array
            featuresNp = features.data.cpu().numpy()
            # Accumulate the results in the means array
            np.add.at(self.means,target, featuresNp)
            # Keep track of how many instances of a class have been seen. This should be an array with all elements = classSize
            np.add.at(self.totalFeatures, target, 1)

        # Divide the means array with total number of instaces to get the average
        self.means=self.means/self.totalFeatures
        # Compute and divide by the L2 norm
        # self.norms = np.linalg.norm(self.means, axis=1)

        # Reshape for broadcasting, and convert into a pytorch tensor.
        # self.means = self.means/self.norms.reshape((self.norms.size,1))
        self.means = torch.from_numpy(self.means).unsqueeze(0)
        print ("Mean vectors computed")
        # Return
        return