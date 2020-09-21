# HyperNet
Pytorch Implementation of "Fully Hyperbolic Convolutional Neural Networks". 

## Usage

### Training
    # Create optimizer for just the classifier and loss
    optimizer = get_optim(net.classifier)
    optimizer.zero_grad()

    # Forward pass thru the network 
    with torch.no_grad():
        YN, Yo = net(images)

    # Compute loss with local graph for classifier + loss
    YN.requires_grad=True
    S = net.classifier(YN)
    loss = misfit(S, labels)

    # Back prop thru classifier
    loss.backward()
    dYN = YN.grad.data.clone().detach()

    # Update weights and clear grad for local graph
    optimizer.step()
    clear_grad(optimizer)

    # Back prop thru network
    with torch.no_grad():
        Y, Yo = net.backward(YN, Yo, dYN, get_optim)