import torchvision
from torchvision import datasets, transforms
import numpy
import torch

# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, classes, name, labels_per_class_train, labels_per_class_test):
        self.classes = classes
        self.name = name
        self.train_data = None
        self.test_data = None
        self.labels_per_class_train = labels_per_class_train
        self.labels_per_class_test = labels_per_class_test


class MNIST(Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self):
        super().__init__(10, "MNIST", 6000, 1000)

        self.train_transform = transforms.Compose(
            [torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
             transforms.RandomCrop(32, padding=4),transforms.Scale(32),
             transforms.ToTensor()])

        self.test_transform = transforms.Compose(
            [transforms.Scale(32), transforms.ToTensor()])

        self.train_data = datasets.MNIST("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.MNIST("data", train=False, transform=self.test_transform, download=True)

    def get_random_instance(self):
        instance = torch.from_numpy(numpy.random.uniform(low=0, high=1, size=(32, 32))).float()
        instance.unsqueeze_(0)
        return instance

class CIFAR100(Dataset):
    def __init__(self):
        super().__init__(100, "CIFAR100", 500, 100)

        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(),])

        self.train_data = datasets.CIFAR100("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.CIFAR100("data", train=False, transform=self.test_transform, download=True)

    def get_random_instance(self):
        instance = torch.from_numpy(numpy.random.uniform(low=0, high=1, size=(3, 32, 32))).float()
        return instance

class CIFAR10(Dataset):
    def __init__(self):
        super().__init__(10, "CIFAR10", 5000, 1000)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),])

        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(),])

        self.train_data = datasets.CIFAR10("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.CIFAR10("data", train=False, transform=self.test_transform, download=True)

    def get_random_instance(self):
        instance = torch.from_numpy(numpy.random.uniform(low=0, high=1, size=(3, 32, 32))).float()
        return instance
