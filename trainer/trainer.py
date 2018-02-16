from __future__ import print_function
import torch.utils.data as td
import torch
import torch.nn.functional as F
from torch.autograd import Variable



#
# class trainer():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ng
#     def __init__(self, model):
#         self.model = model

    ## Training code. To be moved to the trainer class  ## Training code. To be moved to the trainer class
def train(epoch, optimizer, train_loader, leftover, model ,modelFixed, args, verbose=False):
    model.train()
    # print ("Len",int(train_loader.dataset.len/args.batch_size))

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if not args.oversampling:
        # print (train_loader.dataset.weights)
        train_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                   sampler=torch.utils.data.sampler.WeightedRandomSampler(
                                                       train_loader.dataset.weights.tolist(),
                                                       int(train_loader.dataset.len)), batch_size=args.batch_size,
                                                   **kwargs)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        weightVector = (target * 0).int()
        for elem in leftover:
            weightVector = weightVector + (target == elem).int()

        oldClassesIndices = torch.squeeze(torch.nonzero((weightVector > 0)).long())
        newClassesIndices = torch.squeeze(torch.nonzero((weightVector == 0)).long())
        optimizer.zero_grad()
        targetTemp = target

        # Before incremental learing (without any distillation loss.)


        # print ("Weights = ", weights)
        if len(oldClassesIndices) == 0:
            dataOldClasses = data[newClassesIndices]
            targetTemp = target
            targetsOldClasses = target[newClassesIndices]
            target2 = targetsOldClasses
            dataOldClasses, target = Variable(dataOldClasses), Variable(targetsOldClasses)

            output = model(dataOldClasses)
            y_onehot = torch.FloatTensor(len(dataOldClasses), args.classes)
            if args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target2.unsqueeze_(1)
            y_onehot.scatter_(1, target2, 1)
            loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss.backward()
            optimizer.step()

        # After first increment. With distillation loss.
        elif args.no_distill:
            weights = data
            targetDis2 = targetTemp

            y_onehot = torch.FloatTensor(len(data), args.classes)
            if args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            targetDis2.unsqueeze_(1)
            y_onehot.scatter_(1, targetDis2, 1)

            output = model(Variable(data))
            # y_onehot[weightVectorDis] = outpu2.data
            #
            loss = F.binary_cross_entropy(output, Variable(y_onehot))
            loss.backward()
            optimizer.step()
        else:
            # optimizer.zero_grad()
            dataDis = Variable(data[oldClassesIndices])
            targetDis2 = targetTemp

            y_onehot = torch.FloatTensor(len(data), args.classes)
            if args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            targetDis2.unsqueeze_(1)
            y_onehot.scatter_(1, targetDis2, 1)

            outpu2 = modelFixed(dataDis)
            output = model(Variable(data))
            y_onehot[oldClassesIndices] = outpu2.data
            #
            loss = F.binary_cross_entropy(output, Variable(y_onehot))

            loss.backward()
            optimizer.step()




def test(epoch, loader, model, args):
    model.eval()
    test_loss = 0
    correct = 0


    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()



    test_loss /= len(loader.dataset)
    return 100. * correct / len(loader.dataset)
