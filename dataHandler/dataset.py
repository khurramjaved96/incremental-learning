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

        self.train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.test_transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.train_data = datasets.MNIST("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.MNIST("data", train=False, transform=self.test_transform, download=True)


class CIFAR100(Dataset):
    def __init__(self):
        super().__init__(100, "CIFAR100", 500, 100)

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        self.train_transform = transforms.Compose(
             #[torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
             #transforms.RandomCrop(32, padding=6), torchvision.transforms.RandomRotation((-10, 10)),
             [transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.train_data = datasets.CIFAR100("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.CIFAR100("data", train=False, transform=self.test_transform, download=True)

class CIFAR10(Dataset):
    def __init__(self):
        super().__init__(10, "CIFAR10", 5000, 1000)

        #self.train_transform = transforms.Compose(
        #     #[torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
        #     #transforms.RandomCrop(32, padding=6), torchvision.transforms.RandomRotation((-10, 10)),
        #     [transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

       # self.test_transform = transforms.Compose(
       #     [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.alt_transform = transforms.Compose(
             [transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


        self.train_data = datasets.CIFAR10("data", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.CIFAR10("data", train=False, transform=self.test_transform, download=True)
