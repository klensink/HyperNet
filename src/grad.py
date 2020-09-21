import torch
from torch.utils.cpp_extension import load
from torch.nn.modules.utils import _triple

# load the PyTorch extension
cudnn_convolution = load(name="cudnn_convolution", sources=["src/cudnn_convolution.cpp"], verbose=True)

def conv3d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
    r"""
    Computes the gradient of conv3d with respect to the weight of the convolution.
    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    Examples::
        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, weight, grad_output)
        >>> F.grad.conv3d_weight(input, weight.shape, grad_output)
    """
    assert not input.device.type == 'cpu', 'Tensor must be on GPU'

    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    grad_weight = cudnn_convolution.convolution_backward_weight(
        input, 
        weight_size, 
        grad_output, 
        stride, 
        padding, 
        dilation, 
        groups, 
        False, # Benchmark
        False  # Deterministic
    )

    return grad_weight

if __name__ == "__main__":

    # create dummy input, convolutional weights and bias
    input  = torch.zeros(1, 3, 8, 16, 16).to('cuda')
    weight = torch.zeros(3, 3, 3, 3, 3).to('cuda')
    bias   = torch.zeros(3).to('cuda')

    # create dummy gradient w.r.t. the output
    grad_output = torch.rand_like(input)

    # compute the gradient w.r.t. the weights and input
    grad_weight = conv3d_weight(input, weight.shape, grad_output, stride=1, padding=1)