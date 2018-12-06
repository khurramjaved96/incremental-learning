''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import copy
import logging

import numpy as np
import torch
import torch.utils.data as td
from PIL import Image
from torch.autograd import Variable

logger = logging.getLogger('iCARL')


class IncrementalLoader(td.Dataset):
    def __init__(self, data, labels, class_size, classes, active_classes, transform=None, cuda=False,
                 oversampling=True):
        oversampling = not oversampling
        self.len = class_size * len(active_classes)
        sort_index = np.argsort(labels)
        self.class_size = class_size
        if "torch" in str(type(data)):
            data = data.numpy()
        self.data = data[sort_index]
        labels = np.array(labels)
        self.labels = labels[sort_index]
        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.active_classes = active_classes
        self.limited_classes = {}
        self.total_classes = classes
        self.means = {}
        self.cuda = cuda
        self.weights = np.zeros(self.total_classes * self.class_size)
        self.indices = {}
        self.__class_indices()
        self.over_sampling = oversampling
        # f(label) = new_label. We do this to ensure labels are in increasing order. For example, even if first increment chooses class 1,5,6, the training labels will be 0,1,2
        self.indexMapper = {}
        self.no_transformation = False
        self.transformLabels()

    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((self.labels.size, self.labels.max() + 1))
        b[np.arange(self.labels.size), self.labels] = 1
        self.labels = b

    def __class_indices(self):
        cur = 0
        for temp in range(0, self.total_classes):
            cur_len = len(np.nonzero(np.uint8(self.labels == temp))[0])
            self.indices[temp] = (cur, cur + cur_len)
            cur += cur_len

    def add_class(self, n):
        logger.debug("Adding Class %d", n)
        if n in self.active_classes:
            return
        # Mapping each new added classes to new label in increasing order; we switch the label so that the resulting confusion matrix is always in order
        # regardless of order of classes used for incremental training.
        indices = len(self.indexMapper)
        if not n in self.indexMapper:
            self.indexMapper[n] = indices
        self.active_classes.append(n)
        self.len = self.class_size * len(self.active_classes)
        self.__update_length()

    def __update_length(self):
        '''
        Function to compute length of the active elements of the data. 
        :return: 
        '''

        len_var = 0
        for a in self.active_classes:
            len_var += self.indices[a][1] - self.indices[a][0]
        self.len = len_var

        return

    def limit_class(self, n, k):
        if k == 0:
            logging.warning("Removing class %d", n)
            self.remove_class(n)
            self.__update_length()
            return False
        if k > self.indices[n][1] - self.indices[n][0]:
            k = self.indices[n][1] - self.indices[n][0]
        if n in self.limited_classes:
            self.limited_classes[n] = k
            # Remove this line; this turns off oversampling
            if not self.over_sampling:
                self.indices[n] = (self.indices[n][0], self.indices[n][0] + k)
            self.__update_length()
            return False
        else:
            if not self.over_sampling:
                self.indices[n] = (self.indices[n][0], self.indices[n][0] + k)
            self.limited_classes[n] = k
            self.__update_length()
            return True

    def limit_class_and_sort(self, n, k, model):
        ''' This function should only be called the first time a class is limited. To change the limitation, 
        call the limiClass(self, n, k) function 
        
        :param n: Class to limit
        :param k: No of exemplars to keep 
        :param model: Features extracted from this model for sorting. 
        :return: 
        '''

        if self.limit_class(n, k):
            start = self.indices[n][0]
            end = self.indices[n][1]
            buff = np.zeros(self.data[start:end].shape)
            images = []
            # Get input features of all the images of the class
            for ind in range(start, end):
                img = self.data[ind]
                if "torch" in str(type(img)):
                    img = img.numpy()
                img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)
                images.append(img)
            data_tensor = torch.stack(images)
            if self.cuda:
                data_tensor = data_tensor.cuda()

            # Get features
            features = model.forward(Variable(data_tensor), True)
            features_copy = copy.deepcopy(features.data)
            mean = torch.mean(features, 0, True)
            list_of_selected = []

            # Select exemplars
            for exmp_no in range(0, min(k, self.class_size)):
                if exmp_no > 0:
                    to_add = torch.sum(features_copy[0:exmp_no], dim=0).unsqueeze(0)
                    if self.cuda:
                        to_add = to_add.cuda()
                    features_temp = (features + Variable(to_add)) / (exmp_no + 1) - mean
                else:
                    features_temp = features - mean
                features_norm = torch.norm(features_temp.data, 2, dim=1)
                if self.cuda:
                    features_norm = features_norm.cpu()
                arg_min = np.argmin(features_norm.numpy())
                if arg_min in list_of_selected:
                    assert (False)
                list_of_selected.append(arg_min)
                buff[exmp_no] = self.data[start + arg_min]
                features_copy[exmp_no] = features.data[arg_min]
                features[arg_min] = features[arg_min] + 1000
                # logger.debug("Arg Min: %d", arg_min)
            print("Exmp shape", buff[0:min(k, self.class_size)].shape)
            self.data[start:start + min(k, self.class_size)] = buff[0:min(k, self.class_size)]

        self.__update_length()

    def remove_class(self, n):
        while n in self.active_classes:
            self.active_classes.remove(n)
        self.__update_length()

    def __len__(self):
        return self.len

    def get_start_index(self, n):
        '''
        
        :param n: 
        :return: Returns starting index of classs n
        '''
        return self.indices[n][0]

    def getIndexElem(self, bool):
        self.no_transformation = bool

    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        assert (index < len(self.data))
        assert (index < self.len)

        length = 0
        temp_a = 0
        old_len = 0
        for a in self.active_classes:
            temp_a = a
            old_len = length
            length += self.indices[a][1] - self.indices[a][0]
            if length > index:
                break
        base = self.indices[temp_a][0]
        incre = index - old_len
        if temp_a in self.limited_classes:
            incre = incre % self.limited_classes[temp_a]
        index = base + incre
        img = self.data[index]
        if "torch" in str(type(img)):
            img = img.numpy()
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if not self.labelsNormal[index] in self.active_classes:
            print("Active classes", self.active_classes)
            print("Label ", self.labelsNormal[index])
            assert (False)
        if self.no_transformation:
            return img, index, self.labelsNormal[index]
        return img, self.labels[index], self.labelsNormal[index]
