from trainer.gans.dcgan import DCGAN

class GANFactory():
    @staticmethod
    def get_trainer(gan_type, args, total_classes, train_iterator, fixed_classifier, experiment):
        if gan_type=="dcgan":
            return DCGAN(args, total_classes, train_iterator, fixed_classifier, experiment)
        else:
            print ("Unsupported model, trainer not found.")
            assert(False)
