import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.attention import MultiHeadAttention
from model.utils.ffn import PositionWiseFeedForwardNetworks
from model.utils.res import ResidualConnection
from typing import Union, Callable
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim=embedding_dim, heads=heads)
        self.ffn = PositionWiseFeedForwardNetworks(embedding_dim=embedding_dim, d_ff=d_ff, activation=activation)

        self.residual_1 = ResidualConnection(dropout_rate=dropout_rate)
        self.residual_2 = ResidualConnection(dropout_rate=dropout_rate)

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)

        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool):
        # sub layer 1
        x = self.layer_norm_1(x)
        q = k = v = x

        attention_output, _ = self.multi_head_attention(q, k, v, mask)
        sub_layer_1 = self.residual_1(attention_output, x, training)

        # sub layer 2
        sub_layer_1 = self.layer_norm_2(sub_layer_1)
        ffn_output = self.ffn(sub_layer_1)
        sub_layer_2 = self.residual_2(ffn_output, sub_layer_1, training)

        return sub_layer_2

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(embedding_dim=embedding_dim, heads=heads)
        self.multi_head_attention = MultiHeadAttention(embedding_dim=embedding_dim, heads=heads)
        self.ffn = PositionWiseFeedForwardNetworks(embedding_dim=embedding_dim, d_ff=d_ff, activation=activation)

        self.residual_1 = ResidualConnection(dropout_rate=dropout_rate)
        self.residual_2 = ResidualConnection(dropout_rate=dropout_rate)
        self.residual_3 = ResidualConnection(dropout_rate=dropout_rate)

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)
        self.layer_norm_3 = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)
        
        self.to(device)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, look_ahead_mask: torch.Tensor, padding_mask: torch.Tensor, training: bool):
        # sub layer 1
        x = self.layer_norm_1(x)
        q = k = v = x
        masked_attention_output, _ = self.masked_multi_head_attention(q, k, v, look_ahead_mask)
        sub_layer_1 = self.residual_1(masked_attention_output, x, training)

        # sub layer 2
        sub_layer_1 = self.layer_norm_2(sub_layer_1)
        q = sub_layer_1
        k = v = encoder_output
        attention_output, _ = self.multi_head_attention(q, k, v, padding_mask)
        sub_layer_2 = self.residual_2(attention_output, sub_layer_1, training)

        # sub layer 3
        sub_layer_2 = self.layer_norm_3(sub_layer_2)
        ffn_output = self.ffn(sub_layer_2)
        sub_layer_3 = self.residual_3(ffn_output, sub_layer_2, training)

        return sub_layer_3

