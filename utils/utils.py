import numpy as np
import torch.utils.data as td
from PIL import Image

class incrementalLoaderCifar(td.Dataset):
    def __init__(self, data, labels, classSize, classes, activeClasses,transform=None):
        self.len = classSize*len(activeClasses)
        sortIndex = np.argsort(labels)
        self.classSize = classSize
        self.data = data[sortIndex]
        labels = np.array(labels)
        self.labels = labels[sortIndex]
        self.transform = transform
        self.activeClasses=activeClasses
        self.limitedClasses={}
        self.totalClasses = classes
        self.means = {}

    def addClasses(self, n):
        if n in self.activeClasses:
            return
        self.activeClasses.append(n)
        self.len = self.classSize * len(self.activeClasses)


    def limitClass(self,n,k):
        self.limitedClasses[n] = k

    def removeClass(self, n):
        while n in self.activeClasses:
            self.activeClasses.remove(n)
        self.len = self.classSize * len(self.activeClasses)

    def __len__(self):
        return self.len
    def __getitem__(self, index):
        assert(index<self.len)
        classNo = int(index/self.classSize)
        incre = index%self.classSize
        if self.activeClasses[classNo] in self.limitedClasses:
            incre = incre%self.limitedClasses[self.activeClasses[classNo]]

        base = self.activeClasses[classNo]*self.classSize

        index = base+incre
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

    def sortByImportance(self, model):

        pass
    def getBottlenecks(self):
        pass

def resizeImage(img,factor):
    '''
    
    :param img: 
    :param factor: 
    :return: 
    '''
    img2 = np.zeros(np.array(img.shape)*factor)

    for a in range(0,img.shape[0]):
        for b in range(0,img.shape[1]):
            img2[a*factor:(a+1)*factor, b*factor:(b+1)*factor] = img[a,b]
    return img2