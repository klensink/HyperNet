# hypernet
import math
import numpy as np
import time
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import gradcheck
# from torch.nn import init
import matplotlib.pyplot as plt

from src.utils import byte2mb, mem_report, check_mem, model_size, clear_grad
from src.Haar2D import Haar2D
from src.DoubleSym import DoubleSymLayer2D
from src.parallel import ReversibleDataParallel

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def tdot(A,B):
    """
    Dot product between two tensors
    """
    return torch.dot(A.view(-1),B.view(-1))


class HyperNet(nn.Module):

    def __init__(self, channels_in, nclasses, layers_per_unit, h=1e-1, verbose=False, layer_class = DoubleSymLayer2D, wave=Haar2D, clear_grad=True, classifier_type='conv3', act=F.relu):
        """
        HyperNet(self, channels_in, nclasses, layers_per_unit, h=1e-1, verbose=False, wave=HaarDWT, clear_grad=True, classifier_type='conv', act=F.relu):

        Args:
            channels_in (int): Number of channels of the input tensor
            nclasses (int): Number of class channels needed in the output of the classifier
            layers_per_unit (list): (number_layers_in_unit, pooling_mode) This list of tuples specifies the number of layers in each unit
                and the mode of the pooling operator. Pooling modes can either be 'down', 'up', or None. E.g an 8 layer classifier
                is defined as [(4,'down'), (4,None)] and a shallow encoder-decoder network as [(4,'down'), (4,'up'), (4,None)]
            h (float): Step size
            verbose (bool): Verbosity setting to print the shape of the input through out the forward and backward propogation.
            wave (nn.Module): Reversible pooling operator class. Default 'Haar3d'
            layer_class (nn.Module): Reversible pooling operator class. Default 'DoubleSymLayer3D'
            clear_grad (bool): Clears the stored gradient's after taking an optimization step to save memory.
            classifier_type (str): 'linear' for a linear classifier, or 'conv' for a convolutional classifier, or 'conv3'. Default 'conv3'
        
        Examples:
            net = HyperNet(3, 10, [(4,'down'), (4,'down'), (4,None)], classifier_type='linear)
            net = HyperNet(3, 10, [(4,'down'), (4,'up'), (4,None)], classifier_type='conv')
        """
        super().__init__()

        self.h = h
        self.channels_in = channels_in
        self.nclasses = nclasses
        self.clear_grad=clear_grad
        self.act = act

        growth_factor = 8 if classifier_type == 'conv3' else 4

        # Init Units
        units = []
        wavepools = []
        width = channels_in
        for unit_number, (num_layers, mode) in enumerate(layers_per_unit):
            layers = nn.ModuleList([
                layer_class(width, width, id = (unit_number, i)) for i in range(num_layers)
            ])

            # Set the forward mode of the pooling op
            pool = None
            if mode:
                pool = wave(mode)
                width = width*growth_factor if mode == 'down' else width//growth_factor
                assert width > 0, 'Width must be greater than 0'

            wavepools.append(pool)
            units.append(layers)

        self.units = nn.ModuleList(units)
        self.wavepools = nn.ModuleList(wavepools)

        # channels_out = channels_in
        if classifier_type == 'linear':
            channels_out = channels_in*4**(len(self.units)-1)
            self.classifier = nn.Linear(channels_out, nclasses, bias=False)
        elif classifier_type == 'conv':
            self.classifier = nn.Conv2d(width, nclasses, bias=False, padding=1, kernel_size=3, stride=1)
        elif classifier_type == 'conv3':
            self.classifier = nn.Conv3d(width, nclasses, bias=False, padding=1, kernel_size=3, stride=1)
        elif classifier_type == 'bottleneck':
            self.classifier = BottleNeck(width, nclasses)
        else:
            raise NotImplementedError('This classifier type is not yet implemented', classifier_type)

        # Debugging util
        if verbose:
            def verboseprint(*args):
                """
                Prints arguments as long as verbosity is True, otherwise it does nothing
                """
                for arg in args:
                    print(arg)
        else:
            verboseprint = lambda *a: None #No op

        self.vprint = verboseprint

    def forward(self, Y0):
        self.vprint('Forward\n')

        Yo = Y0
        Y = Y0

        for i_unit, (layers, pool) in enumerate(zip(self.units, self.wavepools)):
            for f in layers:

                self.vprint('Layer %d - %d : %d, %d, %d, %d' % (*f.id, *Y.shape))

                tmp = Y
                Y = 2*Y - Yo + self.h**2 * f(Y)
                Yo = tmp

            # Apply DWT to Y and Yo
            if pool is not None:
                self.vprint('Pool In     : %d, %d, %d, %d' % Y.shape)

                if pool.mode == 'down':
                    Y = pool(Y)
                    Yo = pool(Yo)
                elif pool.mode == 'up':
                    Y = pool.inverse(Y)
                    Yo = pool.inverse(Yo)
                else:
                    raise NotImplementedError('This pooling mode is not implemented', pool.mode)

                self.vprint('Pool Out    : %d, %d, %d, %d' % Y.shape)
            self.vprint('')

        return Y, Yo

    def backward(self, YN, Yo, dY, get_optim, use_local_graph=False):
        """ Compute gradients for the model parameters using the reversible property of the network to 
        re-calculate activation. Gradients are updated in place
        
        Arguments:
            YN {torch.Tensor} -- Final state from the forward pass
            Yo {torch.Tensor} -- Second last state from the forward pass
            dY {torch.Tensor} -- Derivative of YN
            get_optim {function} -- Function that returns an optimizer
        
        Returns:
            torch.Tensor -- Recovery of initial state
            torch.Tensor -- Recovery of second state
        """
        self.vprint('Backward\n')

        with torch.no_grad():
            Y = Yo
            Yo = YN
            dYo = torch.zeros_like(dY)

            for i_unit, (layers, pool) in enumerate(zip(self.units[::-1], self.wavepools[::-1])):

                # Reverse pooling operation
                if pool is not None:
                    self.vprint('Back Pool In     : %d, %d, %d, %d' % Y.shape)

                    # If this pooling op was used to downsample in the fwd pass
                    if pool.mode == 'down':

                        # Apply the adjoint of the forward operation to recover state
                        Y = pool.inverse(Y)
                        Yo = pool.inverse(Yo)

                        # Update gradients
                        dY = pool.forwardBackward(dY)
                        dYo = pool.forwardBackward(dYo)

                    # If this pooling op was used to upsample in the fwd pass
                    elif pool.mode == 'up':

                        # Apply the adjoint of the forward operation to recover state
                        Y = pool(Y)
                        Yo = pool(Yo)

                        # Update gradients
                        dY = pool.inverseBackward(dY)
                        dYo = pool.inverseBackward(dYo)

                    self.vprint('Back Pool Out    : %d, %d, %d, %d' % Y.shape)

                # Loop backward thru the layers of the unit
                for f in layers[::-1]:
                    unit_num, layer_num = f.id
                    self.vprint('Layer %d - %d : %d, %d, %d, %d' % (*f.id, *Y.shape))

                    if use_local_graph:
                        with torch.enable_grad():

                            # Create local graph when when evaluating f
                            Y.requires_grad = True
                            Z = f(Y)

                            # Calc directional derivative in local graph
                            assert dY.shape == Z.shape 
                            d = tdot(dY, Z)
                            d.backward()

                        # Calc derivative of activation
                        for p in f.parameters():
                            p._grad = p._grad * self.h**2
                        dYi = Y.grad.data.clone().detach()
                    else:
                        Z, dYi, dK = f.backward(Y, dY)

                    # Update weights and clear grad
                    Z = Z.detach()                
                    if self.clear_grad:
                        optimizer = get_optim(f)
                        optimizer.step()
                        clear_grad(optimizer)

                    # Calc derivative
                    dYtmp = dY
                    if unit_num == 0 and layer_num == 0: # first layer
                        dY = dY - dYo + (self.h**2)*dYi
                    else:
                        dY = 2*dY - dYo + (self.h**2)*dYi
                    dYo = dYtmp

                    # Recover previous state
                    tmp = Y
                    Y = 2*Y - Yo + self.h**2 * Z
                    Yo = tmp

            if Y.requires_grad:
                Y._grad = dY

        return Y, Yo

    def reverse(self, YN, Yo):
        """ Use the reversible property of the network to re-calculate the input to the forward pass.
        
        Arguments:
            YN {torch.Tensor} -- Final state from the forward pass
            Yo {torch.Tensor} -- Second last state from the forward pass
        
        Returns:
            torch.Tensor -- Recovery of initial state
            torch.Tensor -- Recovery of second state
        """

        Y = Yo
        Yo = YN

        for i_unit, (layers, pool) in enumerate(zip(self.units[::-1], self.wavepools[::-1])):

            # Reverse pooling operation
            if pool is not None:
                self.vprint('Back Pool In     : %d, %d, %d, %d, %d' % Y.shape)

                # If this pooling op was used to downsample in the fwd pass
                if pool.mode == 'down':

                    # Apply the adjoint of the forward operation to recover state
                    Y = pool.inverse(Y)
                    Yo = pool.inverse(Yo)

                # If this pooling op was used to upsample in the fwd pass
                elif pool.mode == 'up':

                    # Apply the adjoint of the forward operation to recover state
                    Y = pool(Y)
                    Yo = pool(Yo)

                self.vprint('Back Pool Out    : %d, %d, %d, %d, %d' % Y.shape)

            # Loop backward thru the layers of the unit
            for f in layers[::-1]:
                self.vprint('Layer %d - %d : %d, %d, %d, %d, %d' % (*f.id, *Y.shape))

                # Create local graph when when evaluating f
                Z = f(Y)

                # Recover previous state
                tmp = Y
                Y = 2*Y - Yo + self.h**2 * Z
                Yo = tmp

        return Y, Yo

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    import utils
    from timeit import default_timer as timer

    num_channels = 4
    num_classes = 2
    layers = [(10, None),]
    net = HyperNet(
        num_channels,
        num_classes,
        layers,
        h=1e-1,
        verbose=False,
        clear_grad=True,
        classifier_type='conv3',
    ).to(device)
    print('\n### Model Statistics')
    print('Model Size: %8.1f mb' % utils.model_size(net))
    print('Number of Parameters: %9d' % utils.num_params(net))
    print(' ')

    nex = 4;
    images = torch.randn((4,num_channels,16,16,16)).to(device)
    fwd_start = timer()
    Y_N, Y_Nm1 = net(images)
    fwd_time = timer() - fwd_start

    dYN = torch.randn_like(Y_N)
    get_optim = lambda net: torch.optim.Adam(net.parameters(), lr=1e-2)

    bwd_start = timer()
    Y0, Y1 = net.backward(Y_N, Y_Nm1, dYN, get_optim,False)
    bwd_time = timer() - bwd_start

    print("fwd time: %1.2f \t bwd time: %1.2f" % (fwd_time, bwd_time))
