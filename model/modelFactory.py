import model.densenet as dn
import model.resnet32 as res
import model.testModel as tm
class modelFactory():
    def __init__(self):
        pass
    def getModel(self, modelType, classes, dataset="CIFAR100"):
        if modelType=="densenet":
            return dn.DenseNet(growthRate=12, depth=40, reduction=0.5,
                        bottleneck=True, nClasses=classes)
        elif modelType=="resnet32":
            if dataset=="MNIST":
                return res.resnet32mnist(classes,1)
            return res.resnet32(classes)
        elif modelType=="resnet20":
            return res.resnet20(classes)
        elif modelType=="resnet44":
            return res.resnet44(classes)
        elif modelType=="test":
            return tm.Net(classes)
