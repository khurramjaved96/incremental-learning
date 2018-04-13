from abc import abstractmethod

class GAN():
    '''
    Base class to inherit GAN trainers from
    '''
    def __init__(self, args):
        self.args = args
        self.G = None
        self.D = None

    @abstractmethod
    def train(self, G, D, active_classes, increment):
        pass
