from torch.utils.data import DataLoader, TensorDataset
import torch
class DataBuilder:
    def build_dataset(self, inp_data: torch.Tensor, out_data: torch.Tensor, batch_size: int = 1, shuffle: bool = False, num_workers:int = 0):
        dataset = TensorDataset(inp_data, out_data)
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return trainloader