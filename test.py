#%%
import torch
import torch.nn as nn
import pickle
# %%
a = torch.rand((5, 12, 512))
# %%
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# %%
a = a.to(device)
# %%
norm = nn.LayerNorm(normalized_shape=512)
# %%
norm(a)
# %%
linear = nn.Linear(in_features=512, out_features=20)
# %%
b = linear(a)
# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=512)
    def forward(self, x):
        x = self.norm(x)
        return x
# %%
net = Net()
# %%
net = net.to(device)
# %%
net(a)
# %%
