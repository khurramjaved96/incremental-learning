from trainer.gans.dcgan import DCGAN
from trainer.gans.cdcgan import CDCGAN
from trainer.gans.wgan import WGAN
from trainer.gans.acgan import ACGAN

class GANFactory():
    @staticmethod
    def get_trainer(gan_type, args, total_classes, train_iterator, fixed_classifier, experiment):
        if gan_type=="dcgan":
            return DCGAN(args, total_classes, train_iterator, fixed_classifier, experiment)
        elif gan_type=="wgan":
            return WGAN(args, total_classes, train_iterator, fixed_classifier, experiment)
        elif gan_type=="cdcgan":
            return CDCGAN(args, total_classes, train_iterator, fixed_classifier, experiment)
        elif gan_type=="acgan":
            return ACGAN(args, total_classes, train_iterator, fixed_classifier, experiment)

        else:
            print ("Unsupported model, trainer not found.")
            assert(False)
