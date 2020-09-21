import gc 
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import argparse

import torch.nn.functional as F
import seaborn as sn

def plot_and_save_cf(cf, args, name=''):

    if args.make_plots:
        class_occurance = np.sum(cf, axis=0, keepdims=True)
        normalized_cf = cf/class_occurance
        sn.heatmap(
            normalized_cf,
            annot=True,
        )
        plt.ylabel('Pred Class')
        plt.xlabel('True Class')
        plt.savefig(os.path.join(args.figs_dir, 'cf_' + name + '.png'))
        plt.close()

    np.save(os.path.join(args.model_dir, 'cf_%s.npy' % name), cf)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_2d(data, save_path=None, show=False):

    for i, (item) in enumerate(data):
        name = item['name']
        d = item['data']
        kwargs = item.get('kwargs', {})
        dn = d.detach().cpu().numpy()

        plt.subplot(1, len(data), i+1) 
        plt.imshow(dn, origin='upper', **kwargs)
        plt.axis('off')
        plt.title(name)
        # plt.colorbar()

    if show:
        plt.show()
    else:
        plt.savefig(save_path, figsize=(20,5), dpi = 300)
        plt.close('all')

def plot_3planes(data, save_path=None, show=False):
    planes = ['axial', 'coronal', 'sagittal']

    for i, item in enumerate(data):
        name = item['name']
        d = item['data']
        kwargs = item.get('kwargs', {})
        dn = d.detach().cpu().numpy()

        for j, plane in enumerate(planes):
            plt.subplot(len(data), len(planes), len(planes)*i + j + 1) 
            plt.imshow(np.take(dn, dn.shape[j]//2, axis=j), origin='upper', **kwargs)
            plt.title('_'.join([name,plane]))
            plt.axis('off')

    if show:
        plt.show()
    else:
        plt.savefig(save_path, figsize=(20,20))
        plt.close('all')

def TV_loss(I, eps=1e-3):

    dIdxsq = (I[:, :, 1:, :, :] - I[:, :, :-1, :, :]).pow(2)
    dIdysq = (I[:, :, :, 1:, :] - I[:, :, :, :-1, :]).pow(2)
    dIdzsq = (I[:, :, :, :, 1:] - I[:, :, :, :, :-1]).pow(2)


    dIdxsq = (dIdxsq[:, :, :, :-1, :-1] + dIdxsq[:, :, :, 1:, :-1] + dIdxsq[:, :, :, :-1, 1:] + dIdxsq[:, :, :, 1:, 1:])/4
    dIdysq = (dIdysq[:, :, :-1, :, :-1] + dIdysq[:, :, 1:, :, :-1] + dIdysq[:, :, :-1, :, 1:] + dIdysq[:, :, 1:, :, 1:])/4
    dIdzsq = (dIdzsq[:, :, :-1, :-1, :] + dIdzsq[:, :, 1:, :-1, :] + dIdzsq[:, :, :-1, 1:, :] + dIdzsq[:, :, 1:, 1:, :])/4

    tv = torch.sqrt(dIdxsq + dIdysq + dIdzsq + eps)

    return tv.mean()


def cf_rv(cf, classid=2):
    G = cf.sum(axis=0)[classid] # Ground Truth
    P = cf.sum(axis=1)[classid] # Pred

    return P/G

def getAccuracy(preds, labels):
    assert preds.shape == labels.shape, "Preds and Labels must be same shape"

    acc = (preds==labels).float().mean().item()

    return acc

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

def tensor_size(x):
    return byte2mb(x.numel()*4)
    
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

def onehot_encode(C, labels):
    one_hot = torch.zeros((C, labels.size(0)), device=labels.device, dtype=torch.long)
    target = one_hot.scatter_(0, labels.unsqueeze(0).data, 1)
    return target

def iou(preds, labels, nClasses, smooth = 1e-3):

    f = torch.zeros(nClasses, device=preds.device)
    N = labels.shape[0]

    preds = preds.view(preds.shape[0], -1)
    labels = labels.view(labels.shape[0], -1)

    # Loop over examples in the batch
    for i, (pred, label) in enumerate(zip(preds, labels)):

        labelled_inds = (label >= 0)
        # Scatter to one hot
        pred = onehot_encode(nClasses, pred)
        if (label < 0).any():
            label = label + 1
            label = onehot_encode(nClasses+1, label)
            label = label[1:]
        else:
            label = onehot_encode(nClasses, label)

        inter = (pred & label)[:, labelled_inds].sum((-1)).float()
        union = (pred | label)[:, labelled_inds].sum((-1)).float()


        f += (inter + smooth)/(union + smooth)

    return f/N



def dataset_normalization_stats(dataset, ex_per_update=10, device='cpu', num_workers=4):
    """
    Calculate the mean and standard deviation for each color channel across all input images in a dataset.
    The mean and standard deviation are often used for data normalization.
​
    Input:
        dataset: torch.utils.data.Dataset class that returns and image and its corresponding label.
        ex_per_update(optional): Number of examples to process before updating status.
​
    Returns:
        mean: Per channel mean pixel value.
        std: Per channel pixel value standard deviation.
    """
 
    batch_size = 1
    n = len(dataset)
    cc = dataset[0][0].shape[0]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    mean = torch.zeros(cc).to(device)
    std = torch.zeros(cc).to(device)
    
    for i, (image, _) in enumerate(loader):

        image = image.to(device)

        # Per image cc mean and std, then take average of batch
        mean += image.view(image.shape[0], cc, -1).mean(-1).mean(0)
        std += image.view(image.shape[0], cc, -1).std(-1).mean(0)
    
        # Print progress
        if i == 0:
            print("Progress:")
        elif i % ex_per_update == 0:
            print('\t%4.1f %%' % (100*i/n))
    
    mean /= n
    std /= n

    return mean, std

def dataset_stats(dataset, n_classes=5, cc=1, ex_per_update=10):

    batch_size = 1
    n = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)

    mean = torch.zeros(cc)
    std = torch.zeros(cc)
    weights = torch.zeros(n_classes+1)

    shapes = np.zeros((n, 3))

    for i, batch in enumerate(loader):
        image = batch['image']
        label = batch['label']

        label = label.view(label.shape[0], -1)
        label = label.squeeze()
        shapes[i] = image.shape[-3:]

        # Per image cc mean and std, then take average of batch
        mean += image.view(image.shape[0], cc, -1).mean(-1).mean(0)
        std += image.view(image.shape[0], cc, -1).std(-1).mean(0)

        # One hot encode the label to expand
        # label = label - mu
        label = label + 1
        label = onehot_encode(n_classes+1, label).view(n_classes+1, -1) 
        n_pixels = label.shape[-1]

        # Sum each channel to get class occurance then scale to image size
        weights += label.sum(-1).float()/n_pixels

        # Print progress
        if i == 0:
            print("Progress:")
        elif i % ex_per_update == 0:
            print('\t%4.1f %%' % (100*i/n))
            print(weights/(i+1))

    weights /= n
    mean /= n
    std /= n

    return mean, std, weights, shapes

if __name__ == "__main__":

    nclasses = 3
    pred = torch.randint(0, 2, size=(2,64,64))
    label = pred.clone()
    label[0, :16, ...] = -1

    print(iou(pred.cuda(), label.cuda(), nclasses))