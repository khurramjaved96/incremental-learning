import torchvision
from torchvision import datasets, transforms


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
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        self.test_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        self.train_data = datasets.MNIST("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.MNIST("data", train=False, transform=self.test_transform, download=True)


class CIFAR100(Dataset):
    def __init__(self):
        super().__init__(100, "CIFAR100", 500, 100)

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        #
        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
             transforms.RandomCrop(32, padding=4), torchvision.transforms.RandomRotation((-, 2)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        self.train_data = datasets.CIFAR100("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.CIFAR100("data", train=False, transform=self.test_transform, download=True)
