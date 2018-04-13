from abc import abstractmethod

import torch
from torch.autograd import Variable

class GAN():
    '''
    Base class to inherit the GAN trainers from
    '''
    def __init__(self, args, total_classes, train_iterator, fixed_classifier, experiment):
        '''
        args: arguments
        total_classes: total classes ._.
        train_iterator: Iterator over examples for GAN
        fixed_classifier: Classifier frozen at current increment
        experiment: Instance of Experiment class
        '''
        self.args = args
        self.experiment = experiment
        self.total_classes = total_classes
        self.train_iterator = train_iterator
        self.fixed_classifier = fixed_classifier
        # Noise for generating examples
        self.fixed_noise = torch.randn(100,100,1,1)

    @abstractmethod
    def train(self, G, D, active_classes, increment):
        '''
        Must implement in derived trainers
        '''
        pass
