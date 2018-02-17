import numpy as np
import torch.utils.data as td
from PIL import Image
import torch
from torch.autograd import Variable
import model.modelFactory as mF
from torchvision import datasets, transforms
import torchvision
import copy

class incrementalLoaderCifar(td.Dataset):
    def __init__(self, data, labels, classSize, classes, activeClasses, transform=None, cuda=False, oversampling=True):

        self.len = classSize * len(activeClasses)
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
        self.weights = np.zeros(self.totalClasses*self.classSize)
        self.overSampling = oversampling

    def addClasses(self, n):
        if n in self.activeClasses:
            return
        self.activeClasses.append(n)
        self.len = self.classSize * len(self.activeClasses)
        self.updateLen()

    def updateLen(self):
        '''
        Function to compute length of the active elements of the data. 
        :return: 
        '''
        # Computing len if no oversampling
        lenVar=0
        for a in self.activeClasses:
            if a in self.limitedClasses:
                self.weights[lenVar:lenVar + min(self.classSize, self.limitedClasses[a])] = 1.0 / float(self.limitedClasses[a])
                if self.classSize > self.limitedClasses[a]:
                    self.weights[lenVar + self.limitedClasses[a]:lenVar + self.classSize] = 0
                lenVar += min(self.classSize, self.limitedClasses[a])

            else:
                self.weights[lenVar:lenVar + self.classSize] = 1.0 / float(self.classSize)
                lenVar+= self.classSize

        self.len = lenVar
        # Computing len if oversampling is turned on.
        if self.overSampling:
            self.len= len(self.activeClasses)*self.classSize
        return

    def preprocessImages(self):
        '''
        Preprocess all images so we don't hv
        :return: 
        '''
        temp = self.data
        self.data = []
        for a in temp:
            img = a
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
            self.data.append(img)
        self.data = np.array(self.data)

    def limitClass(self, n, k):
        if k>self.classSize:
            k = self.classSize
        # self.weights[n * self.classSize:(n + 1) * self.classSize] = 0
        # self.weights[n * self.classSize:n*self.classSize+k] = max(1.0 / float(self.classSize), 1.0/float(k))
        if n in self.limitedClasses:
            self.limitedClasses[n] = k
            # self.weights[n] = max(1,float(self.classSize)/k)
            self.updateLen()
            return False
        else:
            self.limitedClasses[n] = k
            # self.weights[n] = max(1, float(self.classSize) / k)
            self.updateLen()
            return True



    def limitClassAndSort(self, n, k, model):
        ''' This function should only be called the first time a class is limited. To change the limitation, 
        call the limiClass(self, n, k) function 
        
        :param n: Class to limit
        :param k: No of exemplars to keep 
        :param model: Features extracted from this model for sorting. 
        :return: 
        '''

        if self.limitClass(n,k):
            start = self.getStartIndex(n)
            end = start+self.classSize
            buff =  np.zeros(self.data[start:end].shape)
            images = [ ]
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
            for exmp_no in range(0, min(k,self.classSize)):
                if exmp_no>0:
                    toAdd = torch.sum(featuresCopy[0:exmp_no],dim=0).unsqueeze(0)
                    if self.cuda:
                        toAdd = toAdd.cuda()
                    featuresTemp = (features+Variable(toAdd))/(exmp_no+1) - mean
                else:
                    featuresTemp = features - mean
                featuresNorm = torch.norm(featuresTemp.data, 2, dim=1)
                # featuresNorm = featuresTemp.norm(dim=1)
                argMin = np.argmin(featuresNorm.numpy())
                if argMin in listOfSelected:
                    assert(False)
                listOfSelected.append(argMin)
                buff[exmp_no] = self.data[start+argMin]
                featuresCopy[exmp_no] = features.data[argMin]
                # print (featuresCopy[exmp_no])
                features[argMin] = features[argMin] + 1000
            print ("Exmp shape",buff[0:min(k,self.classSize)].shape)
            self.data[start:start+min(k,self.classSize)] = buff[0:min(k,self.classSize)]

        self.updateLen()


    def removeClass(self, n):
        while n in self.activeClasses:
            self.activeClasses.remove(n)
        self.len = self.classSize * len(self.activeClasses)
        self.updateLen()

    def __len__(self):
        return self.len

    def getStartIndex(self,n):
        '''
        
        :param n: 
        :return: Returns starting index of classs n
        '''
        return n*self.classSize

    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        assert(index<self.classSize*self.totalClasses)
        if not self.overSampling:
            len = 0
            old = 0
            for a in self.activeClasses:
                oldLen = len
                if a in self.limitedClasses:
                    len += min(self.classSize, self.limitedClasses[a])
                else:
                    len += self.classSize
                if len>index:
                    break
            base = a*self.classSize
            incre = index - oldLen
        else:
            assert (index < self.len)
            classNo = int(index / self.classSize)
            incre = index % self.classSize
            if self.activeClasses[classNo] in self.limitedClasses:
                incre = incre % self.limitedClasses[self.activeClasses[classNo]]

            base = self.activeClasses[classNo] * self.classSize

        index = base + incre
        img = self.data[index]
        if "torch" in str(type(img)):
            img = img.numpy()
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if not self.labels[index] in self.activeClasses:
            print ("Active classes", self.activeClasses)
            print ("Label ", self.labels[index])
        return img, self.labels[index]

    def sortByImportance(self, algorithm = "Kennard-Stone"):
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
            line = line[(line.find(":")+1):]
            # select instances based on priority
            prioritizedIndices = line.split(",")
            for index in prioritizedIndices:
                sortedData.append(self.data[int(index)])
            # select remaining instances
            for i in range(classNum*self.classSize,(classNum+1)*self.classSize):
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
    trainDatasetFull = incrementalLoaderCifar(train_data.train_data, train_data.train_labels, 500, 100, [],
                                                 transform=train_transform)


    train_loader_full = torch.utils.data.DataLoader(trainDatasetFull,
                                                    batch_size=10, shuffle=True)
    myFactory = mF.modelFactory()
    model = myFactory.getModel("test", 100)

    trainDatasetFull.addClasses(2)
    trainDatasetFull.limitClassAndSort(2,60, model)
