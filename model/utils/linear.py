import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class Linear(nn.Module):
    def __init__(self, embedding_dim: int, length: int, sample_rate: int, duration: float, frame_size: int, hop_length: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=(frame_size//8), out_features=embedding_dim)
        self.linear_2 = nn.Linear(in_features=(((sample_rate*duration)/hop_length))//4, out_features=length)
        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)

        x = x.transpose(-1, -2)

        x = self.linear_2(x)

        x = x.transpose(-1, -2)
        
        x = torch.reshape(x, (x.size(0), x.size(2), x.size(3)))
        return x
