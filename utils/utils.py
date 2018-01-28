import pickle
import numpy as np
import torch.utils.data as td
from PIL import Image

class incrementalLoaderCifar(td.Dataset):
    def __init__(self, data, labels, classSize, classes,transform=None):
        self.len = classSize*classes
        sortIndex = np.argsort(labels)
        self.classSize = classSize
        self.data = data[sortIndex]
        labels = np.array(labels)
        self.labels = labels[sortIndex]
        self.transform = transform

    def addClasses(self, n):
        self.len+= n*self.classSize
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        assert(index<self.len)
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

