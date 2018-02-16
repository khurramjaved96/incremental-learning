from torchvision import datasets, transforms
import dataHandler.incrementalLoaderCifar as dL
import torchvision

class datasetFactory():
    def __init__(self):
        pass
    @staticmethod
    def getDataset(name, args, train=True):
        if name=="MNIST":
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]

            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                 transforms.RandomCrop(32, padding=6), torchvision.transforms.RandomRotation((-10, 10)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

            test_transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=6), transforms.ToTensor(), transforms.Normalize(mean, std)])

            if train:
                train_data = datasets.MNIST("data", train=True, transform=train_transform, download=True)
                trainDatasetFull = dL.incrementalLoaderCifar(train_data.train_data, train_data.train_labels, 6000,
                                                             args.classes, [], transform=train_transform,
                                                             cuda=args.cuda,
                                                             oversampling=args.oversampling)
                return train_data, trainDatasetFull

            else:
                test_data = datasets.MNIST("data", train=False, transform=test_transform, download=True)
                testDataset = dL.incrementalLoaderCifar(test_data.test_data, test_data.test_labels, 1000, args.classes,
                                                        [], transform=test_transform, cuda=args.cuda,
                                                        oversampling=args.oversampling)
                return test_data, testDataset


        elif name=="CIFAR100":
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]

            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                 transforms.RandomCrop(32, padding=6), torchvision.transforms.RandomRotation((-10, 10)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])

            if train:
                train_data = datasets.CIFAR100("data", train=True, transform=train_transform, download=True)
                trainDatasetFull = dL.incrementalLoaderCifar(train_data.train_data, train_data.train_labels, 500,
                                                             args.classes, [], transform=train_transform,
                                                             cuda=args.cuda,
                                                             oversampling=args.oversampling)
                return train_data, trainDatasetFull

            else:
                test_data = datasets.CIFAR100("data", train=False, transform=test_transform, download=True)
                testDataset = dL.incrementalLoaderCifar(test_data.test_data, test_data.test_labels, 100, args.classes,
                                                        [], transform=test_transform, cuda=args.cuda,
                                                        oversampling=args.oversampling)
                return test_data, testDataset