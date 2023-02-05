import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.linear import Linear
from model.utils.position import PositionEncoding
from model.utils.layer import EncoderLayer
from typing import Union, Callable
from model.utils.net import PreNet

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, n: int, length: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]], m: int, channels: int, kernel_size: int | tuple, stride: int | tuple):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)

        self.additional_module = PreNet(m=m, channels=channels, kernel_size=kernel_size, stride=stride)

        self.linear = Linear(embedding_dim=embedding_dim, length=length)

        self.positional_encoding = PositionEncoding()

        self.encoder_layers = [EncoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)]

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.embedding_dim = embedding_dim
        self.length = length

        self.to(device)


    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool):
        x = self.conv_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = F.relu(x)

        x = self.additional_module(x)

        x = self.linear(x)

        x = x + self.positional_encoding.generate_position(embedding_dim=self.embedding_dim, length=self.length).to(device)

        for layer in self.encoder_layers:
            x = layer(x, mask, training)

        x = self.layer_norm(x)

        return x

        