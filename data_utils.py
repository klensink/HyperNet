import torch
import torchvision
from torchvision import transforms

import HyperNet

def get_cifar10(batch_size, k, root = '/scratch/klensink/data'):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/scratch/klensink/data', 
        train=True, 
        download=False, 
        transform=train_transform
    )
    valset = torchvision.datasets.CIFAR10(
        root='/scratch/klensink/data', 
        train=False, 
        download=False, 
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    network_geometry = [
        (16, 'down'),
        (16, 'down'),
        (16, 'down'),
        (16, None),
    ]
    net = HyperNet.HyperNet(
        3, 
        k, 
        network_geometry, 
        h=1e-1, 
        classifier_type='bottleneck', 
        clear_grad=False, 
        act=torch.tanh
    )

    return train_loader, val_loader, net

def get_stl10(batch_size, k, root = '/scratch/klensink/data'):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.STL10(
        root='/scratch/klensink/data', 
        split='unlabeled',
        download=False, 
        transform=train_transform
    )
    valset = torchvision.datasets.STL10(
        root='/scratch/klensink/data', 
        split='test', 
        download=False, 
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    network_geometry = [
        (8, 'down'),
        (8, 'down'),
        (8, 'down'),
        (8, 'down'),
        (8, None),
    ]
    net = HyperNet.HyperNet(
        3, 
        k, 
        network_geometry, 
        h=1e-1, 
        classifier_type='bottleneck', 
        clear_grad=False, 
        act=torch.tanh
    )

    return train_loader, val_loader, net