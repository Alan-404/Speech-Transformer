#%%
import torch
from model.utils.net import PreNet
# %%
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
net = PreNet(m=3, channels=32)
# %%
a = torch.rand((1, 1, 150, 240))
# %%
device = torch.device('cuda')
# %%
a = a.to(device)
covn1 = conv1.to(device)
conv2 = conv2.to(device)
# %%
a = conv1(a)
a = conv2(a)
a = net(a)
# %%
a.size()
# %%
a = torch.flatten(a, start_dim=1)
# %%
a.size()
# %%
