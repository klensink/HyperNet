import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.grad import conv3d_weight
# from grad import conv2d_weight
from src.grad import conv2d_weight

ygrad = None
def store_grad(x):
    global ygrad
    ygrad = x

def drelu(x):
    dx = torch.max(torch.zeros_like(x), torch.sign(x))
    return dx

class DoubleSymLayer2D(nn.Module):

    def __init__(self, channels_in, channels_out, act=F.relu, dact=drelu, id = None, kernel_size=3):
        super().__init__()

        self.id = id
        self.kernel_size = kernel_size
        self.K = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, bias=None, padding=self.kernel_size//2)
        self.norm = tv_norm
        self.dnorm = dtv_norm
        self.act = act
        self.dact = dact

    def forward(self, x):
        y = self.K(x)
        y = self.norm(y)
        y = self.act(y)
        y = -F.conv_transpose2d(y, self.K.weight, bias=self.K.bias, padding=self.kernel_size//2)

        return y

    def backward(self, x, dy):

        # Recompute forward
        y1 = self.K(x)
        y2 = self.norm(y1)
        y_act = self.act(y2)
        dyact = self.dact(y2)
        y = -F.conv_transpose2d(y_act, self.K.weight, bias=self.K.bias, padding=self.kernel_size//2)

        # dx = -K'diag(sigma'(x, K))*K*dy
        # dK = d/dK1(y'*K1'*sigma(K2*x)) + d/dK2(y'*K1'*sigma(K2*x))
        #    = d/dK1(sigma(K2*x)*K1*y) +  y'*K1'*dsigma(K2*x)  * d/dK2 (K2*x)
        #    = sigma(K2*x) d/dK1 (K1*y) + y'*K1'*dsigma(K2*x) * d/dK2(K2*x)
        # (put back K1 = -K, K2 = K)
        #    = - sigma(K*x) d/dK (K*y) - (K*y)'*dsigma(K*x) * d/dK (K*x)

        dx1 = self.K(dy)
        dx2 = dyact*dx1
        dx2 = self.dnorm(y1, dx2)
        dx = -F.conv_transpose2d(dx2, self.K.weight, bias=self.K.bias, padding=self.kernel_size//2)

        dK1 = - conv2d_weight(dy, self.K.weight.shape, y_act, padding=self.kernel_size//2)
        dK2 = - conv2d_weight(x, self.K.weight.shape, dx2, padding=self.kernel_size//2)
        dK = dK1 + dK2

        self.K.weight.grad = dK

        return y, dx, dK


class DoubleSymLayer3D(nn.Module):

    def __init__(self, channels_in, channels_out, act=F.relu, dact=drelu, id = None, kernel_size=3):
        super().__init__()

        self.id = id
        self.kernel_size = kernel_size
        self.K = nn.Conv3d(channels_in, channels_out, kernel_size=kernel_size, bias=None, padding=self.kernel_size//2)
        self.norm = tv_norm
        self.dnorm = dtv_norm
        self.act = act
        self.dact = dact

    def forward(self, x):
        y = self.K(x)
        y = self.norm(y)
        y = self.act(y)
        y = -F.conv_transpose3d(y, self.K.weight, bias=self.K.bias, padding=self.kernel_size//2)

        return y

    def backward(self, x, dy):

        # Recompute forward
        y1 = self.K(x)
        y2 = self.norm(y1)
        y_act = self.act(y2)
        dyact = self.dact(y2)
        y = -F.conv_transpose3d(y_act, self.K.weight, bias=self.K.bias, padding=self.kernel_size//2)

        # dx = -K'diag(sigma'(x, K))*K*dy
        # dK = d/dK1(y'*K1'*sigma(K2*x)) + d/dK2(y'*K1'*sigma(K2*x))
        #    = d/dK1(sigma(K2*x)*K1*y) +  y'*K1'*dsigma(K2*x)  * d/dK2 (K2*x)
        #    = sigma(K2*x) d/dK1 (K1*y) + y'*K1'*dsigma(K2*x) * d/dK2(K2*x)
        # (put back K1 = -K, K2 = K)
        #    = - sigma(K*x) d/dK (K*y) - (K*y)'*dsigma(K*x) * d/dK (K*x)

        dx1 = self.K(dy)
        dx2 = dyact*dx1
        dx2 = self.dnorm(y1, dx2)
        dx = -F.conv_transpose3d(dx2, self.K.weight, bias=self.K.bias, padding=self.kernel_size//2)

        if dy.device.type == 'cpu' or x.device.type == 'cpu':
            dK1 = - F.grad.conv3d_weight(dy, self.K.weight.shape, y_act, padding=self.kernel_size//2)
            dK2 = - F.grad.conv3d_weight(x, self.K.weight.shape, dx2, padding=self.kernel_size//2)
        else:
            dK1 = - conv3d_weight(dy, self.K.weight.shape, y_act, padding=self.kernel_size//2)
            dK2 = - conv3d_weight(x, self.K.weight.shape, dx2, padding=self.kernel_size//2)
        dK = dK1 + dK2

        self.K.weight.grad = dK

        return y, dx, dK

def tv_norm(x, eps=1e-5):
    sigma2 = torch.sum(x.pow(2), dim=1, keepdim=True) + eps
    sigma = sigma2.sqrt()

    x = x/sigma

    return x

def dtv_norm(x, dy, eps=1e-5):

    sigma2 = torch.sum(x.pow(2), dim=1, keepdim=True) + eps
    sigma = sigma2.sqrt()

    # dx = d/dx (dy*(x/sigma)) = dy/sigma - (dy*x.^2)/sigma.^3
    # x = x - torch.mean(x, dim=1, keepdim=True)
    dx = dy/sigma - (x*torch.sum(x*dy, dim=1, keepdim=True))/sigma.pow(3)

    return dx

if __name__ == '__main__':


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Define the input
    (N, C, W, H) = (2,4,16,24)
    x = torch.rand(N, C, W, H, requires_grad=True, device=device)
    c = torch.rand(N, C, W, H).to(device)

    # Build the layer
    f = DoubleSymLayer2D(C, C, F.relu, drelu, kernel_size=3).to(device)

    # Compute gradients with auto grad
    y = f(x)
    y.register_hook(store_grad)
    loss = torch.dot(y.view(-1), c.view(-1))
    loss.backward()

    dx = x.grad.clone()
    dy = c
    dK = f.K.weight.grad.clone()

    # reset gradients
    for p in f.parameters():
        p.grad.fill_(0.0)
    x.grad.fill_(0.0)

    y_mine, dxt, dKt = f.backward(x, dy)

    print('dx acc: ', torch.norm(dxt - dx)/torch.norm(dx))
    print('dK acc: ', torch.norm(dKt - dK)/torch.norm(dK))

    ### Compute gradients by hand
    # c'*f(x, K)
    # dx = df/dx (x, K)^T * c
    # dK = df/dK (x, K)^T * c

    # print(x.grad)
    x_bar = tv_norm(x)
    x_bar.sum().backward()
    xgrad = x.grad.clone()

    dx = dtv_norm(x, torch.ones_like(x))

    print(torch.norm(xgrad - dx)/torch.norm(xgrad))
