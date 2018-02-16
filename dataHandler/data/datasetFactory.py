from torchvision import datasets, transforms

class datasetFactory():
    def __init__(self):
        pass
    @staticmethod
    def getDataset(name, transform, train=True):
        if name=="MNIST":
            return datasets.MNIST("data", train=train, transform=transform,  download=True)
        else:
            return datasets.CIFAR100("data", train=train, transform=transform, download=True)