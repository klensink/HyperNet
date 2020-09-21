import torch
import torch.nn.functional as F
import numpy as np

# Adjoint test
N = 8
y1 = torch.rand(N,1)
v1 = torch.rand(N,1)

# Build Wavelet Array
k = np.array([[1,1], [1, -1]])
e = np.eye(N//2, N//2)
W = torch.tensor(np.kron(e,k), dtype=torch.float)

t1 = torch.mm(v1.t(), torch.mm(W,y1))
t2 = torch.mm(y1.t(), torch.mm(W.t(),v1))

print("Adjoint Test (mmul): ", (t1/t2).item())

N = 8
y = y1.view(1,-1).unsqueeze(0)
v = v1.view(1,-1).unsqueeze(0)

filt_lo = torch.Tensor([[1., 1.]])
filt_hi = torch.Tensor([[1.,-1.]])

# Stack the filters to have 2 output channels
filt = torch.stack((filt_lo, filt_hi), dim = 0)

# Apply conv and permute
mm = F.conv1d(y, filt, stride=2, padding=0).view(-1, 1)
tmp = torch.zeros_like(mm)
tmp[::2] = mm[:N//2, :]
tmp[1::2] = mm[N//2:, :]
t1_conv = torch.mm(v1.t(), tmp)

mm = F.conv1d(v, filt, stride=2, padding=0).view(-1, 1) # W' = W
tmp = torch.zeros_like(mm)
tmp[::2] = mm[:N//2, :]
tmp[1::2] = mm[N//2:, :]
t2_conv = torch.mm(y1.t(), tmp)

print("Adjoint Test (conv): ", (t1_conv/t2_conv).item())