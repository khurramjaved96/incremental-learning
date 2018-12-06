''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import model.resnet32 as res
import model.test_model as tm


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset="CIFAR100"):

        if model_type == "resnet32":
            if dataset == "MNIST":
                return res.resnet32mnist(10)
            elif dataset == "CIFAR10":
                return res.resnet32(10)
            return res.resnet32(100)


        elif model_type == "resnet20":
            if dataset == "MNIST":
                return res.resnet20mnist(10)
            elif dataset == "CIFAR10":
                return res.resnet20(10)
            return res.resnet20(100)

        elif model_type == "resnet10":
            if dataset == "MNIST":
                return res.resnet10mnist(10)
            elif dataset == "CIFAR10":
                return res.resnet20(10)
            return res.resnet20(100)


        elif model_type == "resnet44":
            if dataset == "MNIST":
                print("MNIST Dataset not supported in this model. Try resnet20 or 32")
                assert (False)
            elif dataset == "CIFAR10":
                return res.resnet44(10)
            return res.resnet44(100)


        elif model_type == "test":
            if dataset == "MNIST":
                return tm.Net(10, 1)
            elif dataset == "CIFAR10":
                return tm.Net(10)
            return tm.Net(100)
        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
