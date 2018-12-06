''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name):
        if name == "MNIST":
            return data.MNIST()
        elif name == "CIFAR100":
            return data.CIFAR100()
        elif name == "CIFAR10":
            return data.CIFAR10()
        else:
            print("Unsupported Dataset")
            assert False
