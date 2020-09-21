import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np

def unsqueeze2(x):
    return x.unsqueeze(0).unsqueeze(0)

class Haar2D(nn.Module):

    def __init__(self, mode):
        super().__init__()

        self.mode = mode
        filt_lo = torch.Tensor([1.0, 1.0])/2 # Replace with sqrt(1/2)
        filt_hi = torch.Tensor([1.0,-1.0])/2

        ll = torch.ger(filt_lo, filt_lo)
        lh = torch.ger(filt_lo, filt_hi)
        hl = torch.ger(filt_hi, filt_lo)
        hh = torch.ger(filt_hi, filt_hi)

        self.weight = nn.Parameter(
            torch.stack((ll,lh,hl,hh), dim=0).unsqueeze(1), requires_grad=False
        )
    
    def forward(self, x):

        N,C,H,W = x.shape
        filters = torch.cat([self.weight,] * C, dim=0)

        Y = F.conv2d(x, filters, groups=C, stride=2)
    
        return Y
    
    def forwardBackward(self, dY):

        N,C,H,W = dY.shape
        C = C//4

        filters = torch.cat([self.weight,] * C, dim=0)

        dYnm1 = F.conv_transpose2d(dY, filters, groups=C, stride=2)

        return dYnm1

    def inverse(self, x):

        N,C,H,W = x.shape
        C = C//4

        filters = torch.cat([self.weight,] * C, dim=0)

        Y = F.conv_transpose2d(x, filters, groups=C, stride=2)

        return Y*4 # Need to figure out where this factor comes from. Haar should be 1 -1 not 1/2, should account for this

    def inverseBackward(self, dY):
        N,C,H,W = dY.shape
        filters = torch.cat([self.weight,] * C, dim=0)

        dYnm1 = F.conv2d(dY, filters, groups=C, stride=2)
    
        return dYnm1*4


if __name__ == "__main__":

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    nt = 100
    N,C,H,W = (128,192,4,4)
    dwt = Haar2D().to(device)

    # Test Adjoint
    x_adj = torch.randn(N,C,H,W, device=device)
    y_adj = dwt(x_adj)
    xp_adj = dwt.inverse(y_adj)
    err_adj = (xp_adj - x_adj).norm()/x_adj.norm()
    print('Adjoint test: %4.1e' % err_adj)

    # Forward
    t_fwd = []
    for i in range(nt):
        x_fwd = torch.randn(N,C,H,W, requires_grad=True, device=device)
        start_time = time.time()
        y_fwd = dwt(x_fwd)
        t_fwd.append(time.time() - start_time)
    torch.autograd.backward(y_fwd, grad_tensors=torch.ones_like(y_fwd))

    # Inverse
    t_back = []
    for i in range(nt):
        x_back = torch.rand_like(y_fwd, requires_grad=True, device=device)
        start_time = time.time()
        y_back = dwt.inverse(x_back)
        t_back.append(time.time() - start_time)
    torch.autograd.backward(y_back, grad_tensors=torch.ones_like(y_back))

    with torch.no_grad():

        # ForwardBackward
        t_fwdBack = []
        for i in range(nt):
            x_fwdBack = torch.ones_like(y_fwd, device=device)
            start_time = time.time()
            y_fwdBack = dwt.forwardBackward(x_fwdBack)
            t_fwdBack.append(time.time() - start_time)
        err_fwdBack = (y_fwdBack - x_fwd.grad).norm()/x_fwd.grad.norm()

        # InverseBackward
        t_invBack = []
        for i in range(nt):
            x_invBack = torch.ones_like(y_back, device=device)
            start_time = time.time()
            y_invBack = dwt.inverseBackward(x_invBack)
            t_invBack.append(time.time() - start_time)
        err_invBack = (y_invBack - x_back.grad).norm()/x_back.grad.norm()

    #############
    ### Stats ###
    #############

    print('Forward       Inverse       ForwardBackward     InverseBackward     FwdBack Acc     InvBack Acc')
    print('%8.8f    %8.8f    %8.8f          %8.8f          %6.4e         %6.4e' % (
        np.mean(t_fwd),
        np.mean(t_back),
        np.mean(t_fwdBack),
        np.mean(t_invBack),
        err_fwdBack,
        err_invBack
    ))
    