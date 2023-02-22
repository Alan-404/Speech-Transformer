import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
    dk = torch.tensor(k.size(-1)).type(torch.float32)

    attention_scores = torch.matmul(q, k.transpose(-1, -2))
    attention_scores = attention_scores/torch.sqrt(dk)

    if mask is not None:
        attention_scores += mask*(-1e20)

    attention_weights = torch.softmax(attention_scores, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads

        self.head_samples = self.embedding_dim//self.heads

        self.linear_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.linear_output = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.to(device)

    def split(self, x: torch.Tensor):
        batch_size = x.size(0)
        length = x.size(1)

        x = torch.reshape(x, (batch_size, length, self.heads, self.head_samples))
        x = torch.permute(x, (0, 2, 1, 3))
 
        return x

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        batch_size = q.size(0)
        length = q.size(1)

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        

        q_heads = self.split(qw)
        k_heads = self.split(kw)
        v_heads = self.split(vw)

        attention_output, attention_weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention_output = attention_output.permute((0, 2, 1, 3))
        attention_output = attention_output.reshape((batch_size, length, self.embedding_dim))

        output = self.linear_output(attention_output)

        return output, attention_weights



class TwoDimAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_q = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv_k = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv_v = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

        self.conv_output = nn.Conv2d(in_channels=2*channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

        self.to(device)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        conv_q = self.conv_q(q)
        conv_k = self.conv_k(k)
        conv_v = self.conv_v(v)

        q_freq = conv_q # dim = (batch_size, channel, time, freq)
        k_freq = conv_k
        v_freq = conv_v

        q_time = torch.permute(conv_q, (0, 1, 3, 2)) # dim = (batch_size, channel, freq, time)
        k_time = torch.permute(conv_k, (0, 1, 3, 2))
        v_time = torch.permute(conv_v, (0, 1, 3, 2))

        attention_time, _ = scaled_dot_product_attention(q_time, k_time, v_time)
        attention_freq, _ = scaled_dot_product_attention(q_freq, k_freq, v_freq)

        attention_time = torch.permute(attention_time, (0, 1, 3, 2)) # dim = (batch_size, channel, time, freq)

        attention_2D = torch.concat((attention_time, attention_freq), dim=1) # # dim = (batch_size, 2*channel, time, freq)
        output = self.conv_output(attention_2D)

        return output
