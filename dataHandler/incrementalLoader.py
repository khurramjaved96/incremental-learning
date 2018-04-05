import copy

import numpy as np
import torch
import torch.utils.data as td
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
from skimage.transform import resize
import model.modelFactory as mF


class incrementalLoader(td.Dataset):
    def __init__(self, datasetName, data, labels, classSize, classes, activeClasses, transform=None, cuda=False, oversampling=True, alt_transform=None):
        self.len = classSize * len(activeClasses)
        self.datasetName = datasetName
        sortIndex = np.argsort(labels)
        self.classSize = classSize
        if "torch" in str(type(data)):
            data = data.numpy()
        self.data = data[sortIndex]
        labels = np.array(labels)
        self.labels = labels[sortIndex]
        self.transform = transform
        self.activeClasses = activeClasses
        self.limitedClasses = {}
        self.totalClasses = classes
        self.means = {}
        self.cuda = cuda
        self.weights = np.zeros(self.totalClasses * self.classSize)
        self.classIndices()
        self.transformData()
        self.over_sampling = oversampling
        self.alt_transform = alt_transform
        self.do_alt_transform = False


    def transformData(self):
        '''
        Rescale the dataset to 32x32
        TODO: Complete all the transformations here instead of in __getItem__
        '''
        if not self.datasetName == "MNIST":
            return
        temp_data = np.ndarray([self.data.shape[0], 32, 32])
        self.data = np.expand_dims(self.data, axis=3)
        for i in range(len(self.data)):
            temp_data[i] = transforms.Scale(32)(transforms.ToPILImage()(self.data[i]))
        self.data = temp_data


    def classIndices(self):
        self.indices = {}
        cur = 0
        for temp in range(0, self.totalClasses):
            curLen = len(np.nonzero(np.uint8(self.labels == temp))[0])
            self.indices[temp] = (cur, cur + curLen)
            cur += curLen

    def addClasses(self, n):
        if n in self.activeClasses:
            return
        self.activeClasses.append(n)
        self.len = self.classSize * len(self.activeClasses)
        self.updateLen()

    def replaceData(self, data, k):
        '''
        Code to replace images with GAN generated images
        data: Generated images with values in range [-1,1] and of
              shape [C x W x H]
        k   : Number of images to replace per class
        '''
        print ("Replacing data")
        for a in data:
            nump = data[a].data.squeeze().cpu().numpy()
            #nump = resize(new_data, (k, 28, 28), anti_aliasing=True, preserve_range=True)

            #Converting from [-1,1] range to [0,255] because that is what
            #toTensor transform expects
            nump = (((nump/2) + 0.5) * 255).astype(np.uint8)
            if self.datasetName == "CIFAR100" or self.datasetName == "CIFAR10":
                #TODO I think .transpose or .permute does this in one line?
                nump = np.swapaxes(nump, 1, 3)
                nump = np.swapaxes(nump, 1, 2)
            self.data[self.indices[a][0]:self.indices[a][0]+k] = nump

            if a not in self.activeClasses:
                self.activeClasses.append(a)
            self.limitClass(a, k)


    def updateLen(self):
        '''
        Function to compute length of the active elements of the data. 
        :return: 
        '''
        # Computing len if no oversampling
        # for a in self.activeClasses:
        #     if a in self.limitedClasses:
        #         self.weights[lenVar:lenVar + min(self.classSize, self.limitedClasses[a])] = 1.0 / float(
        #             self.limitedClasses[a])
        #         if self.classSize > self.limitedClasses[a]:
        #             self.weights[lenVar + self.limitedClasses[a]:lenVar + self.classSize] = 0
        #         lenVar += min(self.classSize, self.limitedClasses[a])
        #
        #     else:
        #         self.weights[lenVar:lenVar + self.classSize] = 1.0 / float(self.classSize)
        #         lenVar += self.classSize
        #
        # self.len = lenVar
        # Computing len if oversampling is turned on.

        lenVar = 0
        for a in self.activeClasses:
            lenVar += self.indices[a][1] - self.indices[a][0]
        self.len = lenVar

        return

    def limitClass(self, n, k):
        if k == 0:
            self.remove_class(n)
            print("Removed class", n)
            print("Current classes", self.activeClasses)
            return False
        if k > self.classSize:
            k = self.classSize
        if n in self.limitedClasses:
            self.limitedClasses[n] = k
            # Remove this line; this turns off oversampling
            if not self.over_sampling:
                self.indices[n] = (self.indices[n][0], self.indices[n][0] + k)
            self.updateLen()
            return False
        else:
            if not self.over_sampling:
                self.indices[n] = (self.indices[n][0], self.indices[n][0] + k)
            self.limitedClasses[n] = k
            self.updateLen()
            return True

    def remove_class(self, n):
        while n in self.activeClasses:
            self.activeClasses.remove(n)
        self.updateLen()


    def limitClassAndSort(self, n, k, model):
        ''' This function should only be called the first time a class is limited. To change the limitation, 
        call the limiClass(self, n, k) function 
        
        :param n: Class to limit
        :param k: No of exemplars to keep 
        :param model: Features extracted from this model for sorting. 
        :return: 
        '''

        if self.limitClass(n, k):
            start = self.indices[n][0]
            end = self.indices[n][1]
            buff = np.zeros(self.data[start:end].shape)
            images = []
            # Get input features of all the images of the class
            for ind in range(start, end):
                img = self.data[ind]
                if "torch" in str(type(img)):
                    img = img.numpy()
                img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)
                images.append(img)
            dataTensor = torch.stack(images)
            if self.cuda:
                dataTensor = dataTensor.cuda()

            # Get features
            features = model.forward(Variable(dataTensor), True)
            featuresCopy = copy.deepcopy(features.data)
            mean = torch.mean(features, 0, True)
            listOfSelected = []

            # Select exemplars
            for exmp_no in range(0, min(k, self.classSize)):
                if exmp_no > 0:
                    toAdd = torch.sum(featuresCopy[0:exmp_no], dim=0).unsqueeze(0)
                    if self.cuda:
                        toAdd = toAdd.cuda()
                    featuresTemp = (features + Variable(toAdd)) / (exmp_no + 1) - mean
                else:
                    featuresTemp = features - mean
                featuresNorm = torch.norm(featuresTemp.data, 2, dim=1)
                # featuresNorm = featuresTemp.norm(dim=1)
                if self.cuda:
                    featuresNorm = featuresNorm.cpu()
                argMin = np.argmin(featuresNorm.numpy())
                if argMin in listOfSelected:
                    assert (False)
                listOfSelected.append(argMin)
                buff[exmp_no] = self.data[start + argMin]
                featuresCopy[exmp_no] = features.data[argMin]
                # print (featuresCopy[exmp_no])
                features[argMin] = features[argMin] + 1000
            print("Exmp shape", buff[0:min(k, self.classSize)].shape)
            self.data[start:start + min(k, self.classSize)] = buff[0:min(k, self.classSize)]

        self.updateLen()

    def removeClass(self, n):
        while n in self.activeClasses:
            self.activeClasses.remove(n)
        self.updateLen()

    def __len__(self):
        return self.len

    def getStartIndex(self, n):
        '''
        :param n: 
        :return: Returns starting index of classs n
        '''
        return self.indices[n][0]

    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        assert (index < self.classSize * self.totalClasses)

        len = 0
        tempA = 0
        oldLen = 0
        for a in self.activeClasses:
            tempA = a
            oldLen = len
            len += self.indices[a][1] - self.indices[a][0]
            if len > index:
                break
        base = self.indices[tempA][0]
        incre = index - oldLen
        if tempA in self.limitedClasses:
            incre = incre % self.limitedClasses[tempA]
        index = base + incre
        img = self.data[index]
        if "torch" in str(type(img)):
            img = img.numpy()
        img = Image.fromarray(img)

        #if self.data.shape[0] == 60000:
        if self.datasetName == "MNIST":
            img = np.expand_dims(img, axis=2)

        if (not self.do_alt_transform) and self.transform is not None:
            img = self.transform(img)
        else:
            img = self.alt_transform(img)

        if not self.labels[index] in self.activeClasses:
            print("Active classes", self.activeClasses)
            print("Label ", self.labels[index])
            assert (False)

        return img, self.labels[index]

    def sortByImportance(self, algorithm="Kennard-Stone"):
        if algorithm == "LDIS":
            dataFile = "dataHandler/selectedCIFARIndicesForTrainingDataK1.txt"
        elif algorithm == "Kennard-Stone":
            dataFile = "dataHandler/selectedCIFARIndicesForTrainingDataKenStone.txt"

        # load sorted (training) data indices
        lines = [line.rstrip('\n') for line in open(dataFile)]
        sortedData = []

        # iterate for each class
        h = 0
        classNum = 0
        for line in lines:
            line = line[(line.find(":") + 1):]
            # select instances based on priority
            prioritizedIndices = line.split(",")
            for index in prioritizedIndices:
                sortedData.append(self.data[int(index)])
            # select remaining instances
            for i in range(classNum * self.classSize, (classNum + 1) * self.classSize):
                if str(i) not in prioritizedIndices:
                    sortedData.append(self.data[i])
                    h += 1
            classNum += 1
        self.data = np.concatenate(sortedData).reshape(self.data.shape)

    def getBottlenecks(self):
        pass


if __name__ == "__main__":
    # To do : Remove the hard-coded mean and just compute it once using the data
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
         transforms.RandomCrop(32, padding=6), torchvision.transforms.RandomRotation((-30, 30)), transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    train_data = datasets.CIFAR100("data", train=True, transform=train_transform, download=True)
    trainDatasetFull = incrementalLoader(train_data.train_data, train_data.train_labels, 500, 100, [],
                                         transform=train_transform)

    train_loader_full = torch.utils.data.DataLoader(trainDatasetFull,
                                                    batch_size=10, shuffle=True)
    myFactory = mF.modelFactory()
    model = myFactory.getModel("test", 100)

    trainDatasetFull.addClasses(2)
    trainDatasetFull.limitClassAndSort(2, 60, model)
