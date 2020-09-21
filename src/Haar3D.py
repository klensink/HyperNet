import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np

def unsqueeze2(x):
    return x.unsqueeze(0).unsqueeze(0)

class Haar3D(nn.Module):

    def __init__(self, mode):
        super().__init__()

        self.mode = mode
        filt_lo = torch.Tensor([1.0, 1.0])/2 # Replace with sqrt(1/2)
        filt_hi = torch.Tensor([1.0,-1.0])/2

        lll = filt_lo[:, None, None] * filt_lo[None, :, None] * filt_lo[None, None, :]
        llh = filt_lo[:, None, None] * filt_lo[None, :, None] * filt_hi[None, None, :]
        lhl = filt_lo[:, None, None] * filt_hi[None, :, None] * filt_lo[None, None, :]
        lhh = filt_lo[:, None, None] * filt_hi[None, :, None] * filt_hi[None, None, :]
        hll = filt_hi[:, None, None] * filt_lo[None, :, None] * filt_lo[None, None, :]
        hlh = filt_hi[:, None, None] * filt_lo[None, :, None] * filt_hi[None, None, :]
        hhl = filt_hi[:, None, None] * filt_hi[None, :, None] * filt_lo[None, None, :]
        hhh = filt_hi[:, None, None] * filt_hi[None, :, None] * filt_hi[None, None, :]

        # Stack the filters to have 2 output channels
        self.weight = nn.Parameter(torch.stack((lll, llh, lhl, lhh, hll, hlh, hhl, hhh)).unsqueeze(1))
        self.weight.requires_grad=False
    
    def forward(self, x):

        N,C,D,H,W = x.shape
        filters = torch.cat([self.weight,] * C, dim=0)

        Y = F.conv3d(x, filters, groups=C, stride=2)
    
        return Y
    
    def forwardBackward(self, dY):

        N,C,D,H,W = dY.shape
        C = C//8

        filters = torch.cat([self.weight,] * C, dim=0)

        dYnm1 = F.conv_transpose3d(dY, filters, groups=C, stride=2)

        return dYnm1

    def inverse(self, x):

        N,C,D,H,W = x.shape
        C = C//8

        filters = torch.cat([self.weight,] * C, dim=0)

        Y = F.conv_transpose3d(x, filters, groups=C, stride=2)

        return Y*8 # Need to figure out where this factor comes from. Haar should be 1 -1 not 1/2, should account for this

    def inverseBackward(self, dY):
        N,C,D,H,W = dY.shape
        filters = torch.cat([self.weight,] * C, dim=0)

        dYnm1 = F.conv3d(dY, filters, groups=C, stride=2)
    
        return dYnm1*8


if __name__ == "__main__":

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    nt = 100
    N,C,D, H,W = (1, 3, 4, 8, 16)
    dwt = Haar3D('fwd').to(device)

    # Test Adjoint
    x_adj = torch.randn(N,C,D, H,W, device=device)
    y_adj = dwt(x_adj)
    xp_adj = dwt.inverse(y_adj)
    err_adj = (xp_adj - x_adj).norm()/x_adj.norm()
    print('Adjoint test: %4.1e' % err_adj)

    # Forward
    t_fwd = []
    for i in range(nt):
        x_fwd = torch.randn(N,C,D,H,W, requires_grad=True, device=device)
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

    print('Forward       Inverse       ForwardBackward     InverseBackward     FwdBack Err     InvBack Err')
    print('%8.8f    %8.8f    %8.8f          %8.8f          %6.4e         %6.4e' % (
        np.mean(t_fwd),
        np.mean(t_back),
        np.mean(t_fwdBack),
        np.mean(t_invBack),
        err_fwdBack,
        err_invBack
    ))
    