import time
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch import cuda

from HyperMLP import HyperMLP
from utils import byte2mb, mem_report, check_mem, model_size, clear_grad

torch.set_printoptions(precision=16)

raise NotImplementedError('Outdated')

if __name__ == '__main__':

    # autograd = True
    autograd = False

    # Get MNIST dataloaders
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 32
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    valset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Build the network
    net = HyperMLP(100, 784, 10, h=1e-3, verbose=False).to(device)
    print('Model Size: %6.2f' % model_size(net))

    get_optim = lambda net: torch.optim.SGD(net.parameters(), lr = 1e-1, momentum=0.0)
    misfit = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):

        print('Epoch %d' % epoch)
        acc = []
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):

            # Move to GPU and vectorize
            N = images.shape[0]
            images = images.view(N, -1).to(device)
            labels = labels.to(device)

            if autograd:
                optimizer = get_optim(net)
                optimizer.zero_grad()
                # Forward pass thru the network
                YN, Yo = net(images)

                # classifier + loss
                S = net.W(YN)
                loss = misfit(S, labels)
                loss.backward()
                optimizer.step()
            else:
                # Forward pass thru the network
                with torch.no_grad():
                    YN, Yo = net(images)

                # Setup tmp optimizer
                optimizer = get_optim(net.W)
                optimizer.zero_grad()

                # Local graph for classifier + loss
                YN.requires_grad=True
                S = net.W(YN)
                loss = misfit(S, labels)
                loss.backward()
                dYN = torch.tensor(YN.grad.data)

                # Update weights and clear grad
                optimizer.step()
                clear_grad(optimizer)

                # Back prop thru network
                Y, Yo = net.backward(YN, Yo, dYN, get_optim)

            preds = torch.argmax(S, dim=1)
            batch_acc = (preds==labels).sum().float()/labels.shape[0]
            acc.append(batch_acc)

            if i%100==0:
                if i == 0:
                    print('   Batch   Acc      Loss     Prec         EPS      Alloc      Cached      %ModelAlloc  %ModelCached')
                else:

                    # Calc precision of recovery
                    if not autograd:
                        d = (Y - images).norm()/images.norm()
                    else:
                        d = torch.tensor([0])
                    
                    eps = (100*batch_size)/(time.time() - start_time)

                    print('  %6d,  %6.4f,  %6.4f,  %6.4e,  %6.1f,  %6.3f,  %6.3f,  %6.3f,      %6.3f' % (
                        i,
                        np.mean(acc), 
                        loss.item(), 
                        d.item(), 
                        eps, 
                        byte2mb(cuda.max_memory_allocated()), 
                        byte2mb(cuda.max_memory_cached()), 
                        model_size(net)/byte2mb(cuda.max_memory_allocated()), 
                        model_size(net)/byte2mb(cuda.max_memory_cached())))
                    acc = []
                    start_time = time.time()

                    # cuda.reset_max_memory_allocated()
                    # cuda.reset_max_memory_cached()

        # Val set
        acc = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):

                # Move to GPU and vectorize
                N = images.shape[0]
                images = images.view(N, -1).to(device)
                labels = labels.to(device)

                # Forward pass thru the network
                YN, Yo = net(images)

                # Classify + loss
                S = net.W(YN)
                loss = misfit(S, labels)

                preds = torch.argmax(S, dim=1)
                batch_acc = (preds==labels).sum().float()/labels.shape[0]
                acc.append(batch_acc)

            print('Test Accruacy: %6.4f,    Loss: %6.4f\n' % (np.mean(acc), loss.data.item()))
            acc = []
