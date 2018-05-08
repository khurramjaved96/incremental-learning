import pickle
import numpy as np
import plotter.plotter as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import confusionmeter

#TODO CONFIGURE FOR CIFAR
def normalize_images(images, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    #images.sub_(mean[0]).div_(std[0])
    #No need to return but the var is needed
    return images

def compute_acc(preds, labels):
    '''
    Computes the classification acc
    '''
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

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
    img = cMatrix.value()
    import matplotlib.pyplot as plt

    plt.imshow(img, cmap='plasma', interpolation='nearest')
    #plt.colorbar()
    plt.savefig(path + str(epoch) + ".jpg")
    plt.savefig(path + str(epoch) + ".eps", format='eps')
    plt.gcf().clear()
    return 100. * correct / len(test_loader.dataset)

def get_confusion_matrix_nmc(path, model, loader, size, args, means, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    # Get the confusion matrix object
    cMatrix = confusionmeter.ConfusionMeter(size, True)

    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            means = means.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data, True).unsqueeze(1)
        result = (output.data - means.float())

        result = torch.norm(result, 2, 2)
        # NMC for classification
        _, pred = torch.min(result, 1)
        # Evaluate results
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # Add the results in appropriate places in the matrix.
        cMatrix.add(pred, target.data.view_as(pred))

    test_loss /= len(loader.dataset)
    img = cMatrix.value()
    import matplotlib.pyplot as plt

    plt.imshow(img, cmap='plasma', interpolation='nearest')
    #plt.colorbar()
    plt.savefig(path + str(epoch) + ".jpg")
    plt.savefig(path + str(epoch) + ".eps", format='eps')
    plt.gcf().clear()
    return 100. * correct / len(loader.dataset)

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
    myPlotter = pl.Plotter()

    if not isinstance(y, list):
        print("y must be a list of tuples!")
        assert(False)

    for i in range(len(y)):
        experiment.results[y[i][0]] = [x, y[i][1]]
        myPlotter.plot(x, y[i][1], title=plot_name, legend=y[i][0])

    myPlotter.save_fig(experiment.path + "Overall", num_classes)
    experiment.store_json()

def plotEmbeddings(experiment, embedding_name_pairs, plot_name="", range_val=(-.20,.20)):
    """
    Takes avg across all classes of embedding (axis 0) then plots them
    embedding_name_pairs: Should be of type [("Embedding Name 1", Embedding_vector_1), ....]

    """
    embeddings = []
    labels = []
    for i in embedding_name_pairs:
        embeddings.append(np.mean(i[1].squeeze().cpu().numpy(), 0))
        labels.append(i[0])

    myPlotter = pl.Plotter()
    myPlotter.plot_embeddings(embeddings, labels, range_val, experiment, plot_name)
