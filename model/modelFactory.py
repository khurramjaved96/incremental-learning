import model.densenet as dn
import model.resnet32 as res
import model.testModel as tm
import model.cDCGAN as gan

class modelFactory():
    def __init__(self):
        pass
    def getModel(self, modelType, dataset="CIFAR100"):
        if modelType=="densenet":
            if dataset=="MNIST":
                print ("MNIST dataset not supported in this model. Try resnet20 or 32")
                assert(False)
            return dn.DenseNet(growthRate=12, depth=40, reduction=0.5,
                        bottleneck=True, nClasses=100)

        elif modelType=="resnet32":
            if dataset=="MNIST":
                return res.resnet32mnist(10)
            return res.resnet32(100)


        elif modelType=="resnet20":
            if dataset=="MNIST":
                return res.resnet20mnist(10)
            return res.resnet20(100)


        elif modelType=="resnet44":
            if dataset == "MNIST":
                print("MNIST dataset not supported in this model. Try resnet20 or 32")
                assert (False)
            return res.resnet44(100)


        elif modelType=="test":
            if dataset=="MNIST":
                print ("MNIST dataset not supported in this model. Try resnet20 or 32")
                assert(False)
            return tm.Net(100)

        elif modelType=="cdcgan":
            if dataset=="CIFAR100":
                print("CIFAR100 not supported")
                assert(False)
            G = gan.Generator(128)
            D = gan.Discriminator(128)
            G.weight_init(mean=0.0, std=0.02)
            D.weight_init(mean=0.0, std=0.02)
            return G, D

        else:
            print ("Unsupported model; either implement the model in model/modelFactory or choose a different model")
            assert(False)
