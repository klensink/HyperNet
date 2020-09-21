import torch
import torch.nn as nn
import torchvision.transforms as transforms
# import cvtorchvision.cvtransforms as transforms
import torchnet.meter as meter
import threading
from torch.cuda._utils import _get_device_index
from torch._utils import ExceptionWrapper

import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import copy

def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply_backward(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module.backward` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.
    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module.backward(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs

class ReversibleDataParallel(nn.DataParallel):

    def backward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply_backward(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
    

    def parallel_apply_backward(self, replicas, inputs, kwargs):
        return parallel_apply_backward(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


if __name__ == "__main__":
    from src.models.hypernet import HyperNet

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_channels = 4
    num_classes = 2
    batch_size = 32
    N = 32
    get_optim = lambda net: torch.optim.Adam(net.parameters(), lr = 1e-1)

    # Build a model
    layers = [
        # (4, 'down'),
        (4, None),
        # (4, 'up'),
    ]
    net = HyperNet(
        num_channels, 
        num_classes, 
        layers, 
        h=1e-1, 
        verbose=False, 
        clear_grad=False, 
        classifier_type='conv3', 
    )

    classifier = copy.deepcopy(net.classifier)
    net.classifier = None

    if torch.cuda.device_count() > 1:
        print('Using multi-gpu')
        net = ReversibleDataParallel(net)
    
    classifier.to(device)
    net.to(device)
    optimizer = get_optim(net)

    images = torch.rand(batch_size, num_channels, N, N, N)
    start_time = time.time()
    for i in range(10):
        optimizer.zero_grad()

        # Forward pass without AD
        print('Forward')
        with torch.no_grad():
            Y_N, Y_Nm1 = net(images)

        # Setup tmp optimizer
        print('Local')

        # Local graph for classifier + loss
        Y_N.requires_grad=True
        N,C,D,H,W = Y_N.shape
        S = classifier(Y_N)
        loss = S.mean()
        loss.backward()
        dYN = Y_N.grad.data.clone().detach()

        # Back prop thru network
        print('Backward')
        with torch.no_grad():
            Y0, Y1 = net.backward(Y_N, Y_Nm1, dYN, get_optim, use_local_graph=False)

        optimizer.step()
        print(loss.item())
    print(time.time() - start_time)