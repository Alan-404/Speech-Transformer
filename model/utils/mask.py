import torch
import numpy as np


class MaskGenerator:
    def generate_padding_mask(self, tensor: torch.Tensor):
        return torch.Tensor(tensor == 0).type(torch.float32)[:, np.newaxis, np.newaxis, :]

    def generate_look_ahead_mask(self, length: int):
        return torch.triu(torch.ones((length, length)), diagonal=1)

    def generate_mask(self, tensor: torch.Tensor):
        padding_mask = self.generate_padding_mask(tensor)

        look_ahead_mask = self.generate_look_ahead_mask(tensor.size(1))

        look_ahead_mask = torch.maximum(look_ahead_mask, padding_mask)

        return padding_mask, look_ahead_mask