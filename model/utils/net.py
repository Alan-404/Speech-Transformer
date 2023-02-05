import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.attention import TwoDimAttention
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class PreNet(nn.Module):
    def __init__(self, m: int, channels: int, kernel_size: int | tuple, stride: int | tuple):
        super().__init__()
        self.conv_attention = nn.Conv2d(in_channels=1, out_channels=channels ,kernel_size=kernel_size, stride=stride)
        self.attention_layers = [TwoDimAttention(channels=channels, kernel_size=kernel_size, stride=stride) for _ in range(m)]
        self.conv_audio = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1)

        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.conv_attention(x)
        for layer in self.attention_layers:
            x = layer(x, x, x)
        
        x = self.conv_audio(x)
        return x
