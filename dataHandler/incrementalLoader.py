import copy

import numpy as np
import torch
import torch.utils.data as td
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
from skimage.transform import resize
import model.modelFactory as mF


class IncrementalLoader(td.Dataset):
    def __init__(self, dataset_name, data, labels, class_size, classes, active_classes, transform=None, cuda=False, oversampling=True, alt_transform=None):
        self.len = class_size * len(active_classes)
        self.dataset_name = dataset_name
        sort_index = np.argsort(labels)
        self.class_size = class_size
        if "torch" in str(type(data)):
            data = data.numpy()
        self.data = data[sort_index]
        labels = np.array(labels)
        self.labels = labels[sort_index]
        self.transform = transform
        self.active_classes = active_classes
        self.limited_classes = {}
        self.total_classes = classes
        self.means = {}
        self.cuda = cuda
        self.weights = np.zeros(self.total_classes * self.class_size)
        self.class_indices()
        self.transform_data()
        self.over_sampling = oversampling
        self.alt_transform = alt_transform
        self.do_alt_transform = False


    def transform_data(self):
        '''
        Rescale the dataset to 32x32
        TODO: Complete all the transformations here instead of in __getItem__
        '''
        if not self.dataset_name == "MNIST":
            return
        temp_data = np.ndarray([self.data.shape[0], 32, 32])
        self.data = np.expand_dims(self.data, axis=3)
        for i in range(len(self.data)):
            temp_data[i] = transforms.Scale(32)(transforms.ToPILImage()(self.data[i]))
        self.data = temp_data


    def class_indices(self):
        self.indices = {}
        cur = 0
        for temp in range(0, self.total_classes):
            cur_len = len(np.nonzero(np.uint8(self.labels == temp))[0])
            self.indices[temp] = (cur, cur + cur_len)
            cur += cur_len

    def add_classes(self, n):
        if n in self.active_classes:
            return
        self.active_classes.append(n)
        self.len = self.class_size * len(self.active_classes)
        self.update_len()

    def replace_data(self, data, k):
        '''
        Code to replace images with GAN generated images
        data: Generated images with values in range [-1,1] and of
              shape [C x W x H]
        k   : Number of images to replace per class
        '''
        print ("Replacing data")
        for a in data:
            nump = data[a].data.squeeze().cpu().numpy()
            #nump = resize(new_data, (k, 28, 28), anti_aliasing=True, preserve_range=True)

            #Converting from [-1,1] range to [0,255] because that is what
            #toTensor transform expects
            nump = (((nump/2) + 0.5) * 255).astype(np.uint8)
            if self.dataset_name == "CIFAR100" or self.dataset_name == "CIFAR10":
                #TODO I think .transpose or .permute does this in one line?
                nump = np.swapaxes(nump, 1, 3)
                nump = np.swapaxes(nump, 1, 2)
            self.data[self.indices[a][0]:self.indices[a][0]+k] = nump

            if a not in self.active_classes:
                self.active_classes.append(a)
            self.limit_class(a, k)


    def update_len(self):
        '''
        Function to compute length of the active elements of the data. 
        :return: 
        '''
        # Computing len if no oversampling
        # for a in self.active_classes:
        #     if a in self.limited_classes:
        #         self.weights[len_var:len_var + min(self.class_size, self.limited_classes[a])] = 1.0 / float(
        #             self.limited_classes[a])
        #         if self.class_size > self.limited_classes[a]:
        #             self.weights[len_var + self.limited_classes[a]:len_var + self.class_size] = 0
        #         len_var += min(self.class_size, self.limited_classes[a])
        #
        #     else:
        #         self.weights[len_var:len_var + self.class_size] = 1.0 / float(self.class_size)
        #         len_var += self.class_size
        #
        # self.len = len_var
        # Computing len if oversampling is turned on.

        len_var = 0
        for a in self.active_classes:
            len_var += self.indices[a][1] - self.indices[a][0]
        self.len = len_var

        return

    def limit_class(self, n, k):
        if k == 0:
            self.remove_class(n)
            print("Removed class", n)
            print("Current classes", self.active_classes)
            return False
        if k > self.class_size:
            k = self.class_size
        if n in self.limited_classes:
            self.limited_classes[n] = k
            # Remove this line; this turns off oversampling
            if not self.over_sampling:
                self.indices[n] = (self.indices[n][0], self.indices[n][0] + k)
            self.update_len()
            return False
        else:
            if not self.over_sampling:
                self.indices[n] = (self.indices[n][0], self.indices[n][0] + k)
            self.limited_classes[n] = k
            self.update_len()
            return True

    def remove_class(self, n):
        while n in self.active_classes:
            self.active_classes.remove(n)
        self.update_len()


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
                # features_norm = features_temp.norm(dim=1)
                if self.cuda:
                    features_norm = features_norm.cpu()
                arg_min = np.argmin(features_norm.numpy())
                if arg_min in list_of_selected:
                    assert (False)
                list_of_selected.append(arg_min)
                buff[exmp_no] = self.data[start + arg_min]
                features_copy[exmp_no] = features.data[arg_min]
                # print (features_copy[exmp_no])
                features[arg_min] = features[arg_min] + 1000
            print("Exmp shape", buff[0:min(k, self.class_size)].shape)
            self.data[start:start + min(k, self.class_size)] = buff[0:min(k, self.class_size)]

        self.update_len()

    def __len__(self):
        return self.len

    def get_start_index(self, n):
        '''
        :param n: 
        :return: Returns starting index of classs n
        '''
        return self.indices[n][0]

    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        assert (index < self.class_size * self.total_classes)

        len = 0
        temp_a = 0
        old_len = 0
        for a in self.active_classes:
            temp_a = a
            old_len = len
            len += self.indices[a][1] - self.indices[a][0]
            if len > index:
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

        #if self.data.shape[0] == 60000:
        if self.dataset_name == "MNIST":
            img = np.expand_dims(img, axis=2)

        if (not self.do_alt_transform) and self.transform is not None:
            img = self.transform(img)
        else:
            img = self.alt_transform(img)

        if not self.labels[index] in self.active_classes:
            print("Active classes", self.active_classes)
            print("Label ", self.labels[index])
            assert (False)

        return img, self.labels[index]

    def sort_by_importance(self, algorithm="Kennard-Stone"):
        if algorithm == "LDIS":
            data_file = "dataHandler/selectedCIFARIndicesForTrainingDataK1.txt"
        elif algorithm == "Kennard-Stone":
            data_file = "dataHandler/selectedCIFARIndicesForTrainingDataKenStone.txt"

        # load sorted (training) data indices
        lines = [line.rstrip('\n') for line in open(data_file)]
        sorted_data = []

        # iterate for each class
        h = 0
        class_num = 0
        for line in lines:
            line = line[(line.find(":") + 1):]
            # select instances based on priority
            prioritized_indices = line.split(",")
            for index in prioritized_indices:
                sorted_data.append(self.data[int(index)])
            # select remaining instances
            for i in range(class_num * self.class_size, (class_num + 1) * self.class_size):
                if str(i) not in prioritized_indices:
                    sorted_data.append(self.data[i])
                    h += 1
            class_num += 1
        self.data = np.concatenate(sorted_data).reshape(self.data.shape)

    def get_bottlenecks(self):
        pass


if __name__ == "__main__":
    # To do : Remove the hard-coded mean and just compute it once using the data
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
         transforms.RandomCrop(32, padding=6), torchvision.transforms.RandomRotation((-30, 30)), transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    train_data = datasets.CIFAR100("data", train=True, transform=train_transform, download=True)
    train_dataset_full = IncrementalLoader(train_data.train_data, train_data.train_labels, 500, 100, [],
                                         transform=train_transform)

    train_loader_full = torch.utils.data.DataLoader(train_dataset_full,
                                                    batch_size=10, shuffle=True)
    my_factory = mF.ModelFactory()
    model = my_factory.get_model("test", 100)

    train_dataset_full.add_classes(2)
    train_dataset_full.limit_class_and_sort(2, 60, model)
