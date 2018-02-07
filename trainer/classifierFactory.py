from torchnet.meter import confusionmeter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch

class testFactory():
    def __init__(self):
        pass
    def getTester(self, testType="nmc", cuda=True):
        if testType=="nmc":
            return NearestMeanClassifier(cuda)



class NearestMeanClassifier():
    def __init__(self, cuda):
        self.cuda = cuda
        self.means=np.zeros((100, 64))
        self.totalFeatures = np.zeros((100,1))
    def classify(self, model, test_loader, cuda, verbose=False):
        model.eval()
        test_loss = 0
        correct = 0
        cMatrix = confusionmeter.ConfusionMeter(100, True)

        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data, True)
            print ("Shape of output", output.shape)
            print ("Shape of means", self.means.shape)
            0/0
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            cMatrix.add(pred, target.data.view_as(pred))

        test_loss /= len(test_loader.dataset)
        if verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        return 100. * correct / len(test_loader.dataset)

    def updateMeans(self, model, train_loader):
        self.means*=0
        for batch_id, (data, target) in enumerate(train_loader):
            # print ("Input data shape", data.shape)
            features = model.forward(Variable(data), True)
            # print ("Target", target)
            # print (features.data.numpy().shape)
            featuresNp = features.data.numpy()
            # print (featuresNp[0],"\n\n\n", featuresNp[1],"\n\n\n", featuresNp[2])
            # print (target)
            np.add.at(self.means,target, featuresNp)
            np.add.at(self.totalFeatures, target, 1)
        self.means=self.means/self.totalFeatures
        print (self.means[0])
        print ("First")
        print (self.means[1])


        # for a in dataLoader.activeClasses:
        #     base = a * dataLoader.classSize
        #     if a in dataLoader.limitedClasses:
        #         limit = min(dataLoader.limitedClasses[a], dataLoader.classSize)
        #     else:
        #         limit = dataLoader.classSize
        #     allImages = torch.from_numpy(dataLoader.data[base:base+limit])
        #     print ("Shape of images", allImages.shape)
        #     allFeatures = model.forward(Variable(allImages), True)
        #     self.means.append(np.mean(allFeatures, axis=0))
        # self.means = np.array(self.means)
