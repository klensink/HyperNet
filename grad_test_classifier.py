import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.nn import init
from torchvision import transforms
import torchvision

from HyperNet import HyperNet, set_seed
from utils import byte2mb, mem_report, check_mem, model_size, clear_grad

torch.set_printoptions(precision=16)

raise NotImplementedError('Outdated')

if __name__ == '__main__':

    set_seed(1234)

    # Get MNIST dataloaders
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    epochs = 100
    iter_per_update = 100
    batch_size = 32

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/scratch/klensink/data', train=True, download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Build the network
    n = (4, 'down')
    net = HyperNet(3, 10, [n,n,n,n], h=1e-3, verbose=True, clear_grad=False, classifier_type='linear').to(device)
    print('Model Size: %6.2f' % model_size(net))

    get_optim = lambda net: torch.optim.SGD(net.parameters(), lr = 1e-1, momentum=0.0)
    misfit = nn.CrossEntropyLoss()

    # Move to GPU and vectorize
    images, labels = next(iter(train_loader))
    N = images.shape[0]
    images = images.to(device)
    labels = labels.to(device)


    ###################################################################
    ### First pass a batch thru and perform manual grad calculation ###
    ###################################################################

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
    loss.backward()
    dYN = torch.tensor(YN.grad.data)

    # Back prop thru network
    Y, Yo = net.backward(YN, Yo, dYN, get_optim)

    manual_gradients = []
    for p in net.parameters():
        if p.grad is not None:
            manual_gradients.append(torch.tensor(p.grad))

    ###################################################################
    ### Pass the same batch thru and get gradients from autograd    ###
    ###################################################################

    optimizer = get_optim(net)
    optimizer.zero_grad()

    # Forward pass thru the network
    YN, Yo = net(images)

    # classifier + loss
    N,C,H,W = YN.shape
    x = F.avg_pool2d(YN, H).view(N, -1)
    S = net.classifier(x)
    loss = misfit(S, labels)
    loss.backward()

    auto_gradients = []
    for p in net.parameters():
        if p.grad is not None:
            auto_gradients.append(p.grad)

    #############################################
    ### Calculate distances between gradients ###
    #############################################

    param_names = [name for name,param in net.named_parameters()]
    D = [((ag-mg).norm()/ag.norm()).item() for ag,mg in zip(auto_gradients, manual_gradients)]

    print('\n%-32s: %s' % ("Parameter Name", "Error"))
    print('---')
    for name,d in zip(param_names, D):
        print('%-32s: %6.4e' % (name, d))
    
    original = images
    recovery = Y
    res = original - recovery
    d = res.norm()/original.norm()
    print("\nInput Recovery: %6.4e" % d.item())

    plt.subplot(1,3,1)
    plt.imshow(np.moveaxis(images[0].detach().cpu().numpy(), 0, -1))
    plt.title('Original')

    plt.subplot(1,3,2)
    plt.imshow(np.moveaxis(images[0].detach().cpu().numpy(), 0, -1))
    plt.title('Recovery')

    plt.subplot(1,3,3)
    plt.imshow(np.clip(np.moveaxis((images - Y)[0].detach().cpu().numpy(), 0, -1), 0 ,1))
    plt.title('Residual')
    # plt.show()
