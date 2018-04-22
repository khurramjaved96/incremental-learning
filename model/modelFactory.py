import model.densenet as dn
import model.resnet32 as res
import model.testModel as tm
import model.cDCGAN as cdcgan
import model.DCGAN as dcgan
import model.WGAN as wgan
import model.ACGAN as acgan

class ModelFactory():
    def __init__(self):
        pass
    def get_model(self, model_type, dataset="CIFAR100", use_mbd=False, d=64):
        if model_type=="densenet":
            if dataset=="MNIST":
                print ("MNIST dataset not supported in this model. Try resnet20 or 32")
                assert(False)
            return dn.DenseNet(growth_rate=12, depth=40, reduction=0.5,
                        bottleneck=True, n_classes=100)

        elif model_type=="resnet32":
            if dataset=="MNIST":
                return res.resnet32mnist(10)
            elif dataset=="CIFAR10":
                return res.resnet32(10)
            return res.resnet32(100)


        elif model_type=="resnet20":
            if dataset=="MNIST":
                return res.resnet20mnist(10)
            return res.resnet20(100)


        elif model_type=="resnet44":
            if dataset == "MNIST":
                print("MNIST dataset not supported in this model. Try resnet20 or 32")
                assert (False)
            return res.resnet44(100)


        elif model_type=="test":
            if dataset=="MNIST":
                print ("MNIST dataset not supported in this model. Try resnet20 or 32")
                assert(False)
            return tm.Net(100)

        elif model_type=="cdcgan":
            if dataset=="CIFAR100":
                G = cdcgan.Generator(d, 3, 100)
                D = cdcgan.Discriminator(d, 3, 100, use_mbd)
            elif dataset=="CIFAR10":
                G = cdcgan.Generator(d, 3, 10)
                D = cdcgan.Discriminator(d, 3, 10, use_mbd)
            else:
                G = cdcgan.Generator(d)
                D = cdcgan.Discriminator(d, 1, 10, use_mbd)
            G.init_weights(mean=0.0, std=0.02)
            D.init_weights(mean=0.0, std=0.02)
            return G, D

        elif model_type=="dcgan":
            if dataset=="CIFAR100" or dataset=="CIFAR10":
                G = dcgan.Generator(d, 3)
                D = dcgan.Discriminator(d, 3)
            else:
                G = dcgan.Generator(d)
                D = dcgan.Discriminator(d)
            G.init_weights(mean=0.0, std=0.02)
            D.init_weights(mean=0.0, std=0.02)
            return G, D

        elif model_type=="wgan":
            if dataset=="CIFAR100" or dataset=="CIFAR10":
                G = wgan.Generator(d, 3)
                D = wgan.Discriminator(d, 3)
            else:
                G = wgan.Generator(d)
                D = wgan.Discriminator(d)
            G.init_weights(mean=0.0, std=0.02)
            D.init_weights(mean=0.0, std=0.02)
            return G, D

        elif model_type=="acgan":
            num_classes = 100 if dataset=="CIFAR100" else 10
            gen_d = 384
            if d < 16:
                print("[!!!] d<16, You sure??")
                assert False
            if d == 32:
                gen_d = 768
            if dataset=="CIFAR100" or dataset=="CIFAR10":
                G = acgan.Generator(gen_d, 3, num_classes)
                D = acgan.Discriminator(d, 3, num_classes)
            else:
                G = acgan.Generator(gen_d, 1, num_classes)
                D = acgan.Discriminator(d, 1, num_classes)
            G.init_weights(mean=0.0, std=0.02)
            D.init_weights(mean=0.0, std=0.02)
            return G, D

        else:
            print ("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert(False)
