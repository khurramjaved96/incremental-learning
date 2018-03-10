import model.densenet as dn
import model.resnet32 as res
import model.testModel as tm
import model.cDCGAN as cdcgan
import model.DCGAN as dcgan
import model.WGAN as wgan

class modelFactory():
    def __init__(self):
        pass
    def getModel(self, modelType, dataset="CIFAR100", use_mbd=False):
        if modelType=="densenet":
            if dataset=="MNIST":
                print ("MNIST dataset not supported in this model. Try resnet20 or 32")
                assert(False)
            return dn.DenseNet(growthRate=12, depth=40, reduction=0.5,
                        bottleneck=True, nClasses=100)

        elif modelType=="resnet32":
            if dataset=="MNIST":
                return res.resnet32mnist(10)
            elif dataset=="CIFAR10":
                return res.resnet32(10)
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
                G = cdcgan.Generator(128, 3, 100)
                D = cdcgan.Discriminator(128, 3, 100, use_mbd)
            elif dataset=="CIFAR10":
                G = cdcgan.Generator(128, 3, 10)
                D = cdcgan.Discriminator(128, 3, 10, use_mbd)
            else:
                G = cdcgan.Generator(128)
                D = cdcgan.Discriminator(128, 1, 10, use_mbd)
            G.init_weights(mean=0.0, std=0.02)
            D.init_weights(mean=0.0, std=0.02)
            return G, D

        elif modelType=="dcgan":
            if dataset=="CIFAR100":
                G = dcgan.Generator(128, 3)
                D = dcgan.Discriminator(128, 3)
            else:
                G = dcgan.Generator(128)
                D = dcgan.Discriminator(128)
            G.init_weights(mean=0.0, std=0.02)
            D.init_weights(mean=0.0, std=0.02)
            return G, D

        elif modelType=="wgan":
            if dataset=="CIFAR100":
                G = wgan.Generator(128, 3)
                D = wgan.Discriminator(128, 3)
            else:
                G = wgan.Generator(128)
                D = wgan.Discriminator(128)
            G.init_weights(mean=0.0, std=0.02)
            D.init_weights(mean=0.0, std=0.02)
            return G, D

        else:
            print ("Unsupported model; either implement the model in model/modelFactory or choose a different model")
            assert(False)
