import numpy as np
import torch.utils.data as td
from PIL import Image


class incrementalLoaderCifar(td.Dataset):
    def __init__(self, data, labels, classSize, classes, activeClasses, transform=None):
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

    def removeClass(self, n):
        while n in self.activeClasses:
            self.activeClasses.remove(n)
        self.len = self.classSize * len(self.activeClasses)

    def __len__(self):
        return self.len

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


