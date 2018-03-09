import dataHandler.dataset as data


class datasetFactory():
    def __init__(self):
        pass

    @staticmethod
    def getDataset(name):
        if name == "MNIST":
            return data.MNIST()
        elif name == "CIFAR100":
            return data.CIFAR100()
        elif name == "CIFAR10":
            return data.CIFAR10()
        else:
            print("Unsupported dataset")
            assert (False)
