import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import cuda
import matplotlib.pyplot as plt

from HyperNet import HyperNet, set_seed
from utils import byte2mb, mem_report, check_mem, model_size, clear_grad

torch.set_printoptions(precision=16)
set_seed(1234)

raise NotImplementedError('Outdated')

if __name__ == '__main__':

    # autograd = True
    autograd = False

    # Get MNIST dataloaders
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    iter_per_update = 10
    batch_size = 128

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/scratch/klensink/data', train=True, download=False, transform=train_transform)
    valset = torchvision.datasets.CIFAR10(root='/scratch/klensink/data', train=False, download=False, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Build the network
    n = 4
    net = HyperNet(3, 10, [n,n,n,n], h=1, verbose=False, clear_grad=False).to(device)
    print('Model Size: %6.2f' % model_size(net))

    get_optim = lambda net: torch.optim.SGD(net.parameters(), lr = 1e-2, momentum=0.0, weight_decay=5e-4)
    optimizer = get_optim(net) if autograd else None
    misfit = nn.CrossEntropyLoss()

    eps_fwd = []
    eps_back = []
    
    for epoch in range(epochs):

        print('Epoch %d' % epoch)
        acc = []
        start_time_back = time.time()
        for i, (images, labels) in enumerate(train_loader):

            # Move to GPU and vectorize
            N = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)

            if autograd:
                optimizer.zero_grad()
                start_time = time.time()

                # Forward pass thru the network
                YN, Yo = net(images)
                N,C,H,W = YN.shape
                x = F.avg_pool2d(YN, H).view(N, -1)
                S = net.classifier(x)
                loss = misfit(S, labels)
                batch_eps_fwd = N/(time.time() - start_time)

                start_time = time.time()
                loss.backward()
                batch_eps_back = N/(time.time() - start_time)

                optimizer.step()

            else:
                start_time = time.time()

                # Forward pass thru the network
                with torch.no_grad():
                    YN, Yo = net(images)
                
                # Setup tmp optimizer
                optimizer = get_optim(net.classifier)
                optimizer.zero_grad()

                # Local graph for classifier + loss
                YN.requires_grad=True
                N,C,H,W = YN.shape
                x = F.avg_pool2d(YN, H).view(N, -1)
                S = net.classifier(x)
                loss = misfit(S, labels)
                batch_eps_fwd = N/(time.time() - start_time)

                # Back prop thru classifier
                start_time = time.time()
                loss.backward()
                dYN = YN.grad.data.clone().detach()

                # Update weights and clear grad
                optimizer.step()
                clear_grad(optimizer)

                # Back prop thru network
                Y, Yo = net.backward(YN, Yo, dYN, get_optim)
                batch_eps_back = N/(time.time() - start_time)

                # raise Exception(batch_eps_fwd, batch_eps_back)

            preds = torch.argmax(S, dim=1)
            batch_acc = (preds==labels).sum().float()/labels.shape[0]
            acc.append(batch_acc.item())
            eps_back.append(batch_eps_back)
            eps_fwd.append(batch_eps_fwd)

            if i%iter_per_update==0:
                # Compare EPS fwd vs EPS backward
                if i == 0:
                    print('   Batch   Acc      Loss     Prec         EPS Fwd   EPS Back      Alloc     Cached     %ModelAlloc   %ModelCached')
                else:

                    # Calc precision of recovery
                    if not autograd:
                        d = (Y - images).norm()/images.norm()
                    else:
                        d = torch.tensor([0])
                    
                    print('  %6d,  %6.4f,  %6.4f,  %6.4e,  %6.1f,   %6.1f,       %6.3f,   %6.3f,   %6.3f,       %6.3f' % (
                        i,
                        np.mean(acc), 
                        loss.item(), 
                        d.item(), 
                        np.mean(eps_fwd), 
                        np.mean(eps_back), 
                        byte2mb(cuda.max_memory_allocated()), 
                        byte2mb(cuda.max_memory_cached()), 
                        model_size(net)/byte2mb(cuda.max_memory_allocated()), 
                        model_size(net)/byte2mb(cuda.max_memory_cached())))
                    acc = []
                    start_time = time.time()
                    eps_back = []
                    eps_fwd = []

                    # cuda.reset_max_memory_allocated()
                    # cuda.reset_max_memory_cached()

        # Val set
        acc = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):

                # Move to GPU and vectorize
                N = images.shape[0]
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass thru the network
                YN, Yo = net(images)

                # Classify + loss
                N,C,H,W = YN.shape
                x = F.avg_pool2d(YN, H).view(N, -1)
                S = net.classifier(x)
                loss = misfit(S, labels)

                preds = torch.argmax(S, dim=1)
                batch_acc = (preds==labels).sum().float()/labels.shape[0]
                acc.append(batch_acc.item())

            print('Test Accruacy: %6.4f,    Loss: %6.4f\n' % (np.mean(acc), loss.data.item()))
