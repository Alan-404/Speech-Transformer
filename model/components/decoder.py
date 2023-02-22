import torch
import torch.nn as nn
from model.utils.position import PositionalEncoding
from model.utils.layer import DecoderLayer
from typing import Union, Callable
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, n: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.positional_encoding = PositionalEncoding(embedding_dim=embedding_dim)

        self.decoder_layers = [DecoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)]

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

        self.to(device)

    def forward(self,x: torch.Tensor, encoder_output: torch.Tensor, look_ahead_mask: torch.Tensor, padding_mask: torch.Tensor, training: bool):

        x = self.embedding_layer(x)

        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, look_ahead_mask, padding_mask, training)

        x = self.layer_norm(x)
        x = self.linear(x)

        return x