import torch
import torch.nn as nn
from typing import Union, Callable

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class PositionWiseFeedForwardNetworks(nn.Module):
    def __init__(self, embedding_dim: int, d_ff: int, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=embedding_dim, out_features=d_ff)
        self.activation = activation
        self.output_layer = nn.Linear(in_features=d_ff, out_features=embedding_dim)

        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x