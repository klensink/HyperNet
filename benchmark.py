import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import cuda
import matplotlib.pyplot as plt

from src.HyperNet import HyperNet, set_seed
from src.utils import byte2mb, mem_report, check_mem, model_size, clear_grad

torch.set_printoptions(precision=16)
set_seed(1234)

if __name__ == '__main__':


    #############
    ### Setup ###
    #############

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    channels_in = 3
    H,W,D = 32,32,32
    nt = 10

    # Build the network
    layers = [(4, 'down'), (4, 'down'), (4, 'up'), (4, 'up'), (4, None)]
    net = HyperNet(channels_in, 10, layers, h=1, verbose=False, clear_grad=False).to(device)
    print('Model Size: %6.2f' % model_size(net))

    get_optim = lambda net: torch.optim.SGD(net.parameters(), lr = 1e-2, momentum=0.0, weight_decay=5e-4)
    optimizer = get_optim(net)
    misfit = nn.MSELoss()

    eps_fwd = []
    eps_back = []

    ##########################
    ### Benchmark ###
    ##########################
    for i in range(nt):

        images = torch.randn(batch_size, channels_in, H,W,D, device=device)
        labels = torch.randn(batch_size, 10, H,W,D, device=device)

        N = images.shape[0]
        images = images.to(device)
        labels = labels.to(device)

        start_time = time.time()
        # Forward pass thru the network
        with torch.no_grad():
            YN, Yo = net(images)
        
        # Setup tmp optimizer
        optimizer = get_optim(net.classifier)
        optimizer.zero_grad()

        # Local graph for classifier + loss
        YN.requires_grad=True
        N,C,H,W,D = YN.shape
        S = net.classifier(YN)
        loss = misfit(S, labels)
        batch_eps_fwd = N/(time.time() - start_time)

        # Back prop thru classifier
        start_time = time.time()
        loss.backward()
        dYN = YN.grad.data.clone().detach()

        # Update weights and clear grad for local graph
        optimizer.step()
        clear_grad(optimizer)

        # Back prop thru network
        with torch.no_grad():
            Y, Yo = net.backward(YN, Yo, dYN, get_optim)
            batch_eps_back = N/(time.time() - start_time)

        eps_fwd.append(batch_eps_fwd)
        eps_back.append(batch_eps_back)

    print('EPS Fwd  : %6.1f' % np.mean(eps_fwd))
    print('EPS Back : %6.1f' % np.mean(eps_back))