import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class PositionEncoding:
    def __generate_angles(self, embedding_dim: int):
        angles = np.arange(embedding_dim)
        angles[0::2] = angles[1::2]
        angles = 1/(np.power(10000, angles/embedding_dim))
        angles = np.expand_dims(angles, axis=0) # dim = (1, embedding_dim)

        return angles

    def __generate_pos_length(self, length: int):
        pos = np.arange(length)
        pos = np.expand_dims(pos, axis=1)
        return pos

    def generate_position(self, embedding_dim: int, length: int):
        pos = self.__generate_pos_length(length=length)
        angles = self.__generate_angles(embedding_dim=embedding_dim)

        angles_pos = np.dot(pos, angles)

        angles_pos[0::2] = np.sin(angles_pos[0::2])
        angles_pos[1::2] = np.cos(angles_pos[1::2])

        angles_pos = np.expand_dims(angles_pos, axis=0)

        return torch.tensor(angles_pos, dtype=torch.float32)
