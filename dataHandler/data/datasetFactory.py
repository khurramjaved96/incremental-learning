from torchvision import datasets, transforms

class datasetFactory():
    def __init__(self):
        pass
    @staticmethod
    def getDataset(name, transform, train=True):
        if name=="MNIST":
            data = datasets.MNIST("data", train=train, transform=transform,  download=True)