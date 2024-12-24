
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models import NetMultiTask
from utils import train, test, YinYangDataset

if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation for multi-task classification of the Yin-Yang dataset.')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='batch size for training (default: 20)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-ewc', default=False, action='store_true',
                        help='if True, network will be trained without elastic weight consolidation (default: False)')
    parser.add_argument('--lamda', default=4, type=float,
                        help='λ sets how important the old task is compared to the new one when using elastic weight consolidation.')
            
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Instantiate the two Yin-Yang datasets
    yinyang_train = YinYangDataset(size=5000, seed=42)
    yinyang_validation = YinYangDataset(size=1000, seed=41)
    yinyang_test = YinYangDataset(size=1000, seed=40)

    yinyang2_train = YinYangDataset(size=5000, seed=42, offset=1)
    yinyang2_validation = YinYangDataset(size=1000, seed=41, offset=1)
    yinyang2_test = YinYangDataset(size=1000, seed=40, offset=1)

    batchsize_train = args.batch_size
    batchsize_eval = len(yinyang_test)

    yinyang_trainloader = torch.utils.data.DataLoader(yinyang_train, batch_size=batchsize_train, shuffle=True)
    yinyang_testloader = torch.utils.data.DataLoader(yinyang_test, batch_size=batchsize_eval, shuffle=False)

    yinyang2_trainloader = torch.utils.data.DataLoader(yinyang2_train, batch_size=batchsize_train, shuffle=True)
    yinyang2_testloader = torch.utils.data.DataLoader(yinyang2_test, batch_size=batchsize_eval, shuffle=False)

    # Create the network
    model = NetMultiTask(lamda=args.lamda)
    if args.cuda:
        torch.cuda.set_device(0)
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    fisher_estimation_sample_size = batchsize_train * 5
    ewc = not args.no_ewc

    if (not ewc):
        model.lamda = 0

    # Train on Task 1
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, criterion, epoch, yinyang_trainloader, args.cuda, args.log_interval)
        acc = test(model, criterion, yinyang_testloader, args.cuda)

    if (ewc):
        print('Consolidating')
        model.consolidate(model.estimate_fisher(yinyang_trainloader, fisher_estimation_sample_size, batch_size=batchsize_train))

    # Train on Task 2
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, criterion, epoch, yinyang2_trainloader, args.cuda, args.log_interval)
        acc = test(model, criterion, yinyang2_testloader, args.cuda)

    # After re-training, compute both accuracies
    print('Post-training Accuracies')
    print('='*80)
    print('Task 1')
    acc = test(model, criterion, yinyang_testloader, args.cuda)
    print('Task 2')
    acc = test(model, criterion, yinyang2_testloader, args.cuda)