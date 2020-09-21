import gc 
import numpy as np

import torch

def byte2mb(x):
    return x*1e-6

def mem_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

def check_mem(report=False):
    mem_alloc = byte2mb(torch.cuda.memory_allocated())
    mem_cached = byte2mb(torch.cuda.memory_cached())

    if report:
        mem_report()
    
    print('Mem Alloc: %6.2f, Mem Cached: %6.2f' % (mem_alloc, mem_cached))

def model_size(net):
    s = 0
    for p in net.parameters():
        s += p.numel()*4
    return byte2mb(s)

def num_params(net):
    n = 0
    for p in net.parameters():
        n += p.numel()
    return n

def clear_grad(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param._grad = None

def apply_dwt(x, op):
    lowered, coeff_list = op(x)

    # Reshape and cat
    coeffs = coeff_list[0]
    N,C,_,H,W = coeffs.shape
    coeffs = coeffs.view(N,-1,H,W)
    x = torch.cat((lowered, coeffs), 1)

    return x

def split_features(x):
    split = x.shape[1]//4
    lowered = x[:, :split, :, :]
    coeffs = x[:, split:, :, :]
    N, _, H, W = coeffs.shape
    coeffs = coeffs.view(N, split, 3, H, W)

    return lowered, coeffs

def apply_idwt(x, coeffs, op):
    x = op((x, [coeffs]))
    return x

def plottable(x, mode='image'):
    if mode=='image':
        out = np.moveaxis(x.cpu().detach().numpy(), 0, -1).squeeze()
    elif mode =='label':
        out = x.cpu().detach().numpy().squeeze()

    return out

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def iou(preds, labels, nClasses, smooth = 1e-6):
    """
    Computers the mean class wise IOU of a batch.

    Input:
        preds    :   NxHxW predictions
        labels   :   NxHxW labels
        nClasses :   Number of classes in the dataset
        smooth (optional): Smoothing term added to the numerator and denominator to avoid 0/0.

    Output:
        iou : nClasses dimensional vector with per class mean IOU
    """

    N,H,W = preds.shape
    onehot = OneHot(nClasses)    
    f = torch.zeros(nClasses)

    # Loop over examples in the batch
    for i, (pred, label) in enumerate(zip(preds, labels)):

        # Scatter to one hot
        pred = onehot(pred.cpu().unsqueeze(0))
        label = onehot(label.cpu().unsqueeze(0))

        inter = (pred & label).sum((-1, -2)).float()
        union = (pred | label).sum((-1, -2)).float()

        f += (inter + smooth)/(union + smooth)

    return f/N

class OneHot:

    def __init__(self, C):
        self.C = C

    def __call__(self, labels):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
        
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size. 
            Each value is an integer representing correct classification.
        C : integer. 
            number of classes in labels.
        
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''

        one_hot = torch.LongTensor(self.C, labels.size(1), labels.size(2)).zero_()
        target = one_hot.scatter_(0, labels.data, 1)
        
        target = torch.autograd.Variable(target)
            
        return target
