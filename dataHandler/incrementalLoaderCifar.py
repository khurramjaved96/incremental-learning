import numpy as np
import torch.utils.data as td
from PIL import Image
import torch
from torch.autograd import Variable
import model.modelFactory as mF
from torchvision import datasets, transforms
import torchvision

class incrementalLoaderCifar(td.Dataset):
    def __init__(self, data, labels, classSize, classes, activeClasses, transform=None, cuda=False):

        self.len = classSize * len(activeClasses)
        sortIndex = np.argsort(labels)
        self.classSize = classSize
        self.data = data[sortIndex]
        labels = np.array(labels)
        self.labels = labels[sortIndex]
        self.transform = transform
        self.activeClasses = activeClasses
        self.limitedClasses = {}
        self.totalClasses = classes
        self.means = {}
        self.cuda = cuda

    def addClasses(self, n):
        if n in self.activeClasses:
            return
        self.activeClasses.append(n)
        self.len = self.classSize * len(self.activeClasses)

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
        self.limitedClasses[n] = k

    def limitClassAndSort(self, n, k, model):
        ''' This function should only be called the first time a class is limited. To change the limitation, 
        call the limiClass(self, n, k) function 
        
        :param n: Class to limit
        :param k: No of exemplars to keep 
        :param model: Features extracted from this model for sorting. 
        :return: 
        '''

        self.limitedClasses[n]=k
        start = self.getStartIndex(n)
        end = start+self.classSize
        buff =  np.zeros(self.data[start:end].shape)
        images = [ ]
        for ind in range(start, end):
            img = self.data[ind]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        dataTensor = torch.stack(images)
        print ("Limit class sort data shape", dataTensor.shape)
        features = model.forward(Variable(dataTensor), True)
        mean = torch.mean(features, 0, True)
        for exmp_no in range(0, min(k,self.classSize)):


            print ("Features shape", features.shape)


            print ("Mean shape", mean.shape)
        0/0


    def removeClass(self, n):
        while n in self.activeClasses:
            self.activeClasses.remove(n)
        self.len = self.classSize * len(self.activeClasses)

    def __len__(self):
        return self.len

    def getStartIndex(self,n):
        '''
        
        :param n: 
        :return: Returns starting index of classs n
        '''
        return n*self.classSize

    def __getitem__(self, index):
        assert (index < self.len)
        classNo = int(index / self.classSize)
        incre = index % self.classSize
        if self.activeClasses[classNo] in self.limitedClasses:
            incre = incre % self.limitedClasses[self.activeClasses[classNo]]

        base = self.activeClasses[classNo] * self.classSize

        index = base + incre
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def sortByImportance(self, model):

        pass

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

    trainDatasetFull.limitClassAndSort(2,40, model)
