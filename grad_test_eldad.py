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

from src.HyperNet import HyperNet, set_seed
from utils import byte2mb, mem_report, check_mem, model_size, clear_grad

torch.set_printoptions(precision=16)


if __name__ == '__main__':

    set_seed(1234)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    num_features = 16
    N = 400
    k = 1
    batch_size = 1

    # Build the network
    n = (4, 'down')
    net = HyperNet(num_features, k, [(4, 'down'), (4, 'up'), (4, None)], h=1e-5, verbose=True, clear_grad=False, classifier_type='conv').to(device)
    print('Model Size: %6.2f' % model_size(net))

    def get_optim(net): return torch.optim.SGD(
        net.parameters(), lr=1e-1, momentum=0.0)
    misfit = nn.MSELoss()

    # Move to GPU and build fake label
    images = torch.rand((batch_size, num_features, N, N))
    images = images.to(device)
    # YN, _ = net(images)
    labels = torch.rand(batch_size, k, N, N).to(device)

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
    YN.requires_grad = True
    N, C, H, W = YN.shape
    S = net.classifier(YN)
    loss = misfit(S, labels)
    loss.backward()
    dYN = torch.tensor(YN.grad.data)

    # Back prop thru network
    Y, Yo = net.backward(YN, Yo, dYN, get_optim, use_local_graph=False)

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
    N, C, H, W = YN.shape
    S = net.classifier(YN)
    loss = misfit(S, labels)
    loss.backward()

    auto_gradients = []
    for p in net.parameters():
        if p.grad is not None:
            auto_gradients.append(p.grad)

    #############################################
    ### Calculate distances between gradients ###
    #############################################

    param_names = [name for name, param in net.named_parameters()]
    D = [((ag-mg).norm()/ag.norm()).item()
         for ag, mg in zip(auto_gradients, manual_gradients)]

    print('\n%-32s: %s' % ("Parameter Name", "Error"))
    print('---')
    for name, d in zip(param_names, D):
        print('%-32s: %6.4e' % (name, d))

    original = images
    recovery = Y
    res = original - recovery
    d = res.norm()/original.norm()
    print("\nInput Recovery: %6.4e" % d.item())

    # plt.subplot(1, 3, 1)
    # plt.imshow(np.moveaxis(images[0].detach().cpu().numpy(), 0, -1))
    # plt.title('Original')

    # plt.subplot(1, 3, 2)
    # plt.imshow(np.moveaxis(images[0].detach().cpu().numpy(), 0, -1))
    # plt.title('Recovery')

    # plt.subplot(1, 3, 3)
    # plt.imshow(
    #     np.clip(np.moveaxis((images - Y)[0].detach().cpu().numpy(), 0, -1), 0, 1))
    # plt.title('Residual')
    # # plt.show()
