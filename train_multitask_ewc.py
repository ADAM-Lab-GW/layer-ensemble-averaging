
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=3,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

from dataset import YinYangDataset

yinyang_train = YinYangDataset(size=5000, seed=42)
yinyang_validation = YinYangDataset(size=1000, seed=41)
yinyang_test = YinYangDataset(size=1000, seed=40)

yinyang2_train = YinYangDataset(size=5000, seed=42, offset=1)
yinyang2_validation = YinYangDataset(size=1000, seed=41, offset=1)
yinyang2_test = YinYangDataset(size=1000, seed=40, offset=1)

batchsize_train = 20
batchsize_eval = len(yinyang_test)

yinyang_trainloader = torch.utils.data.DataLoader(yinyang_train, batch_size=batchsize_train, shuffle=True)
yinyang_testloader = torch.utils.data.DataLoader(yinyang_test, batch_size=batchsize_eval, shuffle=False)

yinyang2_trainloader = torch.utils.data.DataLoader(yinyang2_train, batch_size=batchsize_train, shuffle=True)
yinyang2_testloader = torch.utils.data.DataLoader(yinyang2_test, batch_size=batchsize_eval, shuffle=False)

class NetMNIST(nn.Module):
    def __init__(self):
        super(NetMNIST, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)

class NetMultiTask(nn.Module):
    def __init__(self):
        super(NetMultiTask, self).__init__()

        self.fc1 = nn.Linear(4, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 3)
        self.htanh = nn.Hardtanh()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.layers = [self.fc1, self.fc2, self.fc3]#, self.fc4]
        self.lamda = 4 # λ sets how important the old task is compared to the new one and i labels each parameter.
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)


    def forward(self, x):
        x = x.view(-1, 4)
        x = self.fc1(x)
        x = self.tanh(x)

        x = self.fc2(x)
        x = self.tanh(x)

        x = self.fc3(x)
        # x = self.drop(x)
        return x

    def estimate_fisher(self, data_loader, sample_size, batch_size):
        # sample loglikelihoods from the dataset
        loglikelihoods = []
        for x, y in data_loader:
            x = x.view(batch_size, -1)
            x = x.float()
            x = Variable(x)
            y = Variable(y)
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[torch.autograd.grad(
            l, self.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            self.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

model = NetMultiTask()
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

fisher_estimation_sample_size = batchsize_train * 5
ewc = True

if (not ewc):
    model.lamda = 0

def train(epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        data = data.float()
        optimizer.zero_grad()
        output = model(data)
        ce_loss = criterion(output, target)

        ewc_loss = model.ewc_loss()
        loss = ce_loss + ewc_loss

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            data = data.float()
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)

import numpy as np
max_acc = 0

plot = True

if (plot):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(15, 8))
    titles = ['Training set', 'Test set']
    for i, loader in enumerate([yinyang_trainloader, yinyang_testloader]):
        axes[i].set_title(titles[i])
        axes[i].set_aspect('equal', adjustable='box')
        xs = []
        ys = []
        cs = []
        for batch, batch_labels in loader:
            for j, item in enumerate(batch):
                x1, y1, x2, y2 = item
                c = batch_labels[j]
                xs.append(x1)
                ys.append(y1)
                cs.append(c)
        xs = np.array(xs)
        ys = np.array(ys)
        cs = np.array(cs)
        axes[i].scatter(xs[cs == 0], ys[cs == 0], color='C0', edgecolor='k', alpha=0.7)
        axes[i].scatter(xs[cs == 1], ys[cs == 1], color='C1', edgecolor='k', alpha=0.7)
        axes[i].scatter(xs[cs == 2], ys[cs == 2], color='C2', edgecolor='k', alpha=0.7)
        axes[i].set_xlabel('x1')
        if i == 0:
            axes[i].set_ylabel('y1')


    for i, loader in enumerate([yinyang2_trainloader, yinyang2_testloader]):
        axes[i].set_title(titles[i])
        axes[i].set_aspect('equal', adjustable='box')
        xs = []
        ys = []
        cs = []
        for batch, batch_labels in loader:
            for j, item in enumerate(batch):
                x1, y1, x2, y2 = item
                c = batch_labels[j]
                xs.append(x1)
                ys.append(y1)
                cs.append(c)
        xs = np.array(xs)
        ys = np.array(ys)
        cs = np.array(cs)
        axes[i].scatter(xs[cs == 0], ys[cs == 0], color='C3', edgecolor='k', alpha=0.7)
        axes[i].scatter(xs[cs == 1], ys[cs == 1], color='C4', edgecolor='k', alpha=0.7)
        axes[i].scatter(xs[cs == 2], ys[cs == 2], color='C5', edgecolor='k', alpha=0.7)
        axes[i].set_xlabel('x1')
        if i == 0:
            axes[i].set_ylabel('y1')


    plt.show()
    exit()


# Task 1
for epoch in range(1, args.epochs + 1):
    train(epoch, yinyang_trainloader)
    acc = test(yinyang_testloader)
    if (acc > max_acc):
        max_acc = acc

    print(max_acc)

if (ewc):
    print('Consolidating')
    model.consolidate(model.estimate_fisher(yinyang_trainloader, fisher_estimation_sample_size, batch_size=batchsize_train))

# Task 2 retraining
for epoch in range(1, args.epochs + 1):
    train(epoch, yinyang2_trainloader)
    acc = test(yinyang2_testloader)
    if (acc > max_acc):
        max_acc = acc

    print(max_acc)

# After re-training, compute both accuracies
print('accuracies after retraining')
acc = test(yinyang_testloader)
acc = test(yinyang2_testloader)