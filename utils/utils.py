import pickle
import numpy as np
import plotter.plotter as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import confusionmeter

#TODO CONFIGURE FOR CIFAR
def normalize_images(images, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    #images.sub_(mean[0]).div_(std[0])
    #No need to return but the var is needed
    return images

def get_new_iterator(cuda, train_loader, new_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_iterator = torch.utils.data.DataLoader(train_loader,
                                                batch_size=new_batch_size,
                                                shuffle=True, **kwargs)
    return train_iterator

def resizeImage(img, factor):
    '''
    
    :param img: 
    :param factor: 
    :return: 
    '''
    img2 = np.zeros(np.array(img.shape) * factor)

    for a in range(0, img.shape[0]):
        for b in range(0, img.shape[1]):
            img2[a * factor:(a + 1) * factor, b * factor:(b + 1) * factor] = img[a, b]
    return img2


def saveConfusionMatrix(epoch, path, model, args, dataset, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    cMatrix = confusionmeter.ConfusionMeter(dataset.classes, True)

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if epoch > 0:
            cMatrix.add(pred.squeeze(), target.data.view_as(pred).squeeze())

    test_loss /= len(test_loader.dataset)
    import cv2
    img = cMatrix.value() * 255
    cv2.imwrite(path + str(epoch) + ".jpg", img)
    return 100. * correct / len(test_loader.dataset)


def constructExperimentName(args):
    import os
    name = [args.model_type, str(args.epochs_class), str(args.step_size)]
    if not args.no_herding:
        name.append("herding")
    if not args.no_distill:
        name.append("distillation")
    if not os.path.exists("../" + args.name):
        os.makedirs("../" + args.name)

    return "../" + args.name + "/" + "_".join(name)

# y should be of type [("Value Name", y_values), ....]
def plotAccuracy(experiment, x, y, num_classes, plot_name):
    myPlotter = plt.plotter()

    if not isinstance(y, list):
        print("y must be a list of tuples!")
        assert(False)

    for i in range(len(y)):
        experiment.results[y[i][0]] = [x, y[i][1]]
        myPlotter.plot(x, y[i][1], title=plot_name, legend=y[i][0])

    myPlotter.saveFig(experiment.path + "Overall" + ".jpg", num_classes)

    with open(experiment.path + "result", "wb") as f:
        pickle.dump(experiment, f)
