import torchvision
from torchvision import datasets, transforms


# To incdude a new dataset, inherit from dataset and add all the dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

class dataset():
    '''
    Base class to reprenent a dataset
    '''

    def __init__(self, classes, name, labelsPerClassTrain, labelsPerClassTest):
        self.classes = classes
        self.name = name
        self.trainData = None
        self.testData = None
        self.labelsPerClassTrain = labelsPerClassTrain
        self.labelsPerClassTest = labelsPerClassTest


class MNIST(dataset):
    '''
    Class to include MNIST specific details 
    '''

    def __init__(self):
        super().__init__(10, "MNIST", 6000, 1000)

        self.trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=6),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.testTransform = transforms.Compose(
            [transforms.Scale(32), transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.trainData = datasets.MNIST("data", train=True, transform=self.trainTransform, download=True)

        self.testData = datasets.MNIST("data", train=False, transform=self.testTransform, download=True)


class CIFAR100(dataset):
    def __init__(self):
        super().__init__(100, "CIFAR100", 500, 100)

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        self.trainTransform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
             transforms.RandomCrop(32, padding=6), torchvision.transforms.RandomRotation((-10, 10)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        self.testTransform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        self.trainData = datasets.CIFAR100("data", train=True, transform=self.trainTransform, download=True)

        self.testData = datasets.CIFAR100("data", train=False, transform=self.testTransform, download=True)
