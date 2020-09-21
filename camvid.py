import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchnet.meter as tnt
from torchvision import transforms
from tensorboardX import SummaryWriter

import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from inferno.io.box.camvid import get_camvid_loaders
from PIL import Image

from HyperNet import HyperNet
from utils import bcolors, plottable, iou, model_size, num_params

raise NotImplementedError('Outdated')

class_color = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]

def LabelToPILImage(label):
    # label = label.unsqueeze(0)
    colored_label = torch.zeros(3, label.size(-2), label.size(-1)).byte()
    for i, color in enumerate(class_color):
        mask = label.eq(i)
        for j in range(3):
            colored_label[j].masked_fill_(mask, color[j])
    npimg = colored_label.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    mode = None
    if npimg.shape[2] == 1:
        npimg = npimg[:, :, 0]
        mode = "L"

    return Image.fromarray(npimg, mode=mode)

def getAccuracy(preds, labels):
    assert preds.shape == labels.shape, "Preds and Labels must be same shape"
    N = preds.numel()
    acc = (preds==labels).sum().item()/N

    return acc

def validate(net, K, b, W, misfit, val_loader, device, epoch, cmap, summary_writer=None):

    # For now just test on one image from the training set, later loop over val set
    running_loss = tnt.AverageValueMeter()
    running_acc = tnt.AverageValueMeter()

    for _, (images, labels) in enumerate(val_loader):

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            # Forward Pass
            X = net(images, K, b=b, make_plot=make_plot)
            outputs = conv1x1(X,W)

            loss = misfit(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            acc = getAccuracy(preds, labels)

        running_loss.add(loss.item())
        running_acc.add(acc)

    if summary_writer is not None:
        summary_writer.add_scalar('Val Loss', running_loss.mean, epoch)
        summary_writer.add_scalar('Val Acc', running_acc.mean, epoch)
    
    if True:

        plt.subplot(1,3,1)
        image = images[0]
        image -= image.min()
        image /= image.max()
        plt.imshow(plottable(image, mode='image'))
        plt.title('Image')

        plt.subplot(1,3,2)
        im = LabelToPILImage(labels[0].cpu())
        label_plot = np.array(im)
        plt.imshow(label_plot)
        plt.title('label')

        plt.subplot(1,3,3)
        im = LabelToPILImage(preds[0].cpu())
        label_plot = np.array(im)
        plt.imshow(label_plot)
        plt.title('preds')

        # plt.show()
        # raise Exception()

        plt.savefig('figs/wavnet/val/%06d.png' % epoch)

    print('\n    Validation Loss: %6.2f, Acc: %6.2f' % (running_loss.mean, running_acc.mean*100))

    return running_acc.mean, running_loss.mean

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    batch_size = 4
    train_loader, val_loader, test_loader = get_camvid_loaders(
        '/scratch/klensink/data/CamVid',
        image_shape=(320, 320), 
        # image_shape=(256, 256), 
        labels_as_onehot=False,
        train_batch_size=batch_size, 
        validate_batch_size=batch_size, 
        test_batch_size=batch_size,
        num_workers=8
    )

    train_dataset = train_loader.dataset
    N = len(train_dataset)

    # Test the output of the dataloader
    if False:
        image, label = train_dataset[0]

        image_plot = (image - image.min())
        image_plot /= image_plot.max()
        plt.subplot(1,2,1)
        plt.imshow(np.moveaxis(image_plot.numpy(),0,-1))

        plt.subplot(1,2,2)
        im = LabelToPILImage(label)
        label_plot = np.array(im)
        plt.imshow(label_plot)
        plt.show()

    # PARAMS
    lr=1e-1
    nClasses = len(train_dataset.CLASS_WEIGHTS)
    channels_in = 3
    iter_per_update = 10
    num_epochs = 400
    create_writer=True

    # Create net and init weights
    n=3
    layers = [
        (n,'down'),
        (n,'down'),
        (n,'down'),
        (n,'down'),
        (n,'up'),
        (n,'up'),
        (n,'up'),
        (n,'up'),
        (n, None),
    ]
    net = HyperNet(channels_in, nClasses, layers, h = 1e-2, classifier_type='conv', verbose=False)
    print('Model Size: %6.2f' % model_size(net))
    print('Number of Parameters: %d' % num_params(net))
    print('Number of Layers: %d' % (n*len(layers)))
    net = net.to(device)

    # Show scale captured

    class_weights = torch.tensor(train_dataset.CLASS_WEIGHTS)
    misfit = nn.CrossEntropyLoss(class_weights).to(device)

    best_val_acc = 0
    for epoch in range(num_epochs):

        if epoch%100==0 and not epoch==0:
            lr = lr/10
        get_optim = lambda net: torch.optim.Adam(net.parameters(), lr = lr)

        running_loss = tnt.AverageValueMeter()
        running_acc = tnt.AverageValueMeter()
        running_iou = tnt.AverageValueMeter()

        needs_header = True
        print(bcolors.BOLD + '\n=> Training Epoch #%d | LR %1.2e' %(epoch, lr) + bcolors.ENDC)

        t_fwd = []
        t_back = []
        eps = []
        prec = []

        # Training Loop
        eps_start = time.time()
        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            start_time = time.time()
            with torch.no_grad():
                YN, Yo = net(images)

            # Setup tmp optimizer
            optimizer = get_optim(net.classifier)
            optimizer.zero_grad()

            # Local graph for classifier + loss
            YN.requires_grad=True
            N,C,H,W = YN.shape
            S = net.classifier(YN)
            loss = misfit(S, labels)
            batch_t_fwd = (time.time() - start_time)

            start_time = time.time()
            loss.backward()
            dYN = YN.grad.data.clone().detach()

            # Back prop thru network
            Y0, Y1 = net.backward(YN, Yo, dYN, get_optim)
            batch_t_back = (time.time() - start_time)
            batch_eps = N/(time.time() - eps_start)

            # Keep stats
            preds = torch.argmax(S, dim=1)
            acc = getAccuracy(preds, labels)
            current_iou = iou(preds, labels, nClasses).mean()
            batch_prec = ((Y0 - images).norm()/images.norm()).item()

            running_loss.add(loss.item())
            running_acc.add(acc)
            running_iou.add(current_iou)
            t_fwd.append(batch_t_fwd)
            t_back.append(batch_t_back)
            prec.append(batch_prec)
            eps.append(batch_eps)

            if (batch_idx%iter_per_update==0 and not batch_idx==0):
                end_time = time.time()
                if needs_header:
                    update_hdr = ' ' + bcolors.UNDERLINE + '   Iter     Loss         Acc     IOU       EPS       T Fwd          T Back           Prec' + bcolors.ENDC
                    print(update_hdr)
                    needs_header = False

                batch_loss = running_loss.mean
                batch_acc = running_acc.mean
                batch_iou = running_iou.mean

                total_time = end_time - start_time
                eps = batch_size*iter_per_update/total_time
                update_str = '   %5d     %6.4f     %6.2f    %6.4f    %5.1f     %1.2e       %1.2e         %1.2e' % (
                    batch_idx, 
                    running_loss.mean, 
                    running_acc.mean*100, 
                    running_iou.mean, 
                    np.mean(eps),
                    np.mean(t_fwd),
                    np.mean(t_back),
                    np.mean(prec)
                )

                running_loss = tnt.AverageValueMeter()
                running_acc = tnt.AverageValueMeter()
                running_iou = tnt.AverageValueMeter()
                print(update_str)
                start_time = time.time()
                t_fwd = []
                t_back = []
                eps = []
                prec = []
            
        # Plot a training example
        if True:

            cmap = 'tab20'

            plt.subplot(1,3,1)
            image = images[0]
            image -= image.min()
            image /= image.max()
            plt.imshow(plottable(image, mode='image'))
            plt.title('Image')

            plt.subplot(1,3,2)
            im = LabelToPILImage(labels[0].cpu())
            label_plot = np.array(im)
            plt.imshow(label_plot)
            plt.title('label')

            plt.subplot(1,3,3)
            im = LabelToPILImage(preds[0].cpu())
            label_plot = np.array(im)
            plt.imshow(label_plot)
            plt.title('preds')

            # raise Exception()
            # plt.show()

            plt.savefig('figs/hypernet/train/%06d.png' % epoch)

        # Tensorboard
        if create_writer==True:
            summary_writer = SummaryWriter('log/%s_%s_%s' % ('HyperNet', 'camvid', datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))) 
            create_writer=False
        summary_writer.add_scalar('Train Loss', batch_loss, epoch)
        summary_writer.add_scalar('Train Acc', batch_acc, epoch)


        # # Validate
        # val_acc, val_loss = validate(net, K, b, W, misfit, val_loader, device, epoch, cmap, summary_writer=summary_writer)

        # # Save best model
        # if True and (val_acc > best_val_acc):
        #     print(bcolors.OKGREEN + '    Saving best model %6.4f' % (val_acc*100) + bcolors.ENDC)
        #     best_val_acc = val_acc

        #     model = {
        #         'net':net,
        #         'K':K,
        #         'W':W,
        #         'b':b,
        #     }
        #     torch.save(model, 'models/wavnet/model.ckpt')
