import torch
import torch.autograd as Variable
import torch.nn.functional as F



class trainer():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ng
    def __init__(self, model):
        self.model = model

    ## Training code. To be moved to the trainer class
    def train(epoch, optimizer, train_loader, leftover, args, verbose=False):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if len(weightVectorDis) == 0:
                dataNorm = data[weightVectorNor]
                targetTemp = target
                targetNorm = target[weightVectorNor]
                target2 = targetNorm
                dataNorm, target = Variable(dataNorm), Variable(targetNorm)

                output = self.model(dataNorm)
                y_onehot = torch.FloatTensor(len(dataNorm), args.classes)
                if args.cuda:
                    y_onehot = y_onehot.cuda()

                y_onehot.zero_()
                target2.unsqueeze_(1)
                y_onehot.scatter_(1, target2, 1)
                loss = F.binary_cross_entropy(F.softmax(output), Variable(y_onehot))
                loss.backward()
                optimizer.step()

            # After first increment. With distillation loss.
            elif args.no_distill:
                targetDis2 = targetTemp

                y_onehot = torch.FloatTensor(len(data), args.classes)
                if args.cuda:
                    y_onehot = y_onehot.cuda()

                y_onehot.zero_()
                targetDis2.unsqueeze_(1)
                y_onehot.scatter_(1, targetDis2, 1)

                output = self.model(Variable(data))
                # y_onehot[weightVectorDis] = outpu2.data
                #
                loss = F.binary_cross_entropy(F.softmax(output), Variable(y_onehot))
                loss.backward()
                optimizer.step()
            else:
                # optimizer.zero_grad()
                dataDis = Variable(data[weightVectorDis])
                targetDis2 = targetTemp

                y_onehot = torch.FloatTensor(len(data), args.classes)
                if args.cuda:
                    y_onehot = y_onehot.cuda()

                y_onehot.zero_()
                targetDis2.unsqueeze_(1)
                y_onehot.scatter_(1, targetDis2, 1)

                outpu2 = F.softmax(modelFixed(dataDis))
                output = self.model(Variable(data))
                y_onehot[weightVectorDis] = outpu2.data
                #
                loss = F.binary_cross_entropy(F.softmax(output), Variable(y_onehot))

                loss.backward()
                optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))