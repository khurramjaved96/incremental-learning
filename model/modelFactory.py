import model.densenet as dn
import model.resnet32 as res
import model.testModel as tm
class modelFactory():
    def __init__(self):
        pass
    @staticmethod
    def getModel(modelType, dataset="CIFAR100"):
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
                return tm.testNetMNIST()
            return tm.Net(100)
        else:
            print ("Unsupported model; either implement the model in model/modelFactory or choose a different model")
            assert(False)
