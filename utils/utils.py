import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import confusionmeter


def resize_image(img, factor):
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


def save_confusion_matrix(epoch, path, model, args, dataset, test_loader):
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
            cMatrix.add(pred, target.data.view_as(pred))

    test_loss /= len(test_loader.dataset)
    img = cMatrix.value()

    plt.imshow(img, cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.savefig(path + str(epoch) + ".jpg")
    plt.gcf().clear()
    return 100. * correct / len(test_loader.dataset)


import matplotlib.pyplot as plt

cur = 1


# Function to plot images;



def plot(img, title, g=True):
    global cur, fig
    p = fig.add_subplot(10, 10, cur)
    p.set_title(title)
    cur += 1
    if g:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)


def visualizeTensor(t, path):
    global cur, fig
    cur = 1
    # Function to plot images;
    fig = plt.figure(figsize=(10, 10))
    for a in t:
        img = a.cpu().numpy()
        img = np.swapaxes(img, 0, 2)
        imgMin = np.min(img)

        # img = img-np.min(img)
        # img = img/np.max(img)
        plot(img, str(cur), False)
    plt.savefig(path)
    plt.gcf().clear()
    plt.close()
