import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model.optimizer import ScheduledOptimizer
from typing import Union, Callable
from model.components.encoder import Encoder
from model.components.decoder import Decoder
from model.utils.mask import MaskGenerator

class SpeechTransformerModel(nn.Module):
    def __init__(self, vocab_size: int, 
                n_e: int, 
                n_d: int, 
                embedding_dim: int, 
                length_seq: int, 
                heads: int, 
                d_ff: int, 
                dropout_rate: float, 
                eps: float, 
                activation: Union[str, Callable[[torch.Tensor], torch.Tensor]],
                m: int,
                channels: int,
                kernel_size: int | tuple,
                stride: int | tuple):
        super().__init__()
        self.encoder = Encoder(n=n_e, length=length_seq, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation, m=m, channels=channels, kernel_size=kernel_size, stride=stride)
        self.decoder = Decoder(vocab_size=vocab_size, n=n_d, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)

        self.mask_generator = MaskGenerator()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, encoder_padding_mask: torch.Tensor,decoder_padding_mask: torch.Tensor, look_ahead_mask: torch.Tensor, training: bool):
        
        encoder_output = self.encoder(inputs, encoder_padding_mask, training)
        decoder_output = self.decoder(labels, encoder_output, look_ahead_mask, decoder_padding_mask, training)

        return decoder_output

class SpeechTransformer:
    def __init__(self, 
                vocab_size: int, 
                length_seq: int, 
                n_e: int = 12, 
                n_d: int = 6, 
                embedding_dim: int = 256, 
                heads: int = 4, 
                d_ff: int = 2048, 
                dropout_rate: float = 0.1, 
                eps: float = 0.1, 
                activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                m: int = 2,
                channels: int = 32,
                kernel_size: int | tuple = 3,
                stride: int | tuple = 2):
        self.model = SpeechTransformerModel(vocab_size, n_e, n_d, embedding_dim, length_seq, heads, d_ff, dropout_rate, eps, activation, m, channels, kernel_size, stride)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.mask_generator = MaskGenerator()

    def build_dataset(self, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int):
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def loss_function(self, outputs: torch.Tensor, labels: torch.Tensor):
        length = labels.size(1)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        for item in range(length):
            loss = criterion(outputs[:, item, :], labels[:, item])
            total_loss += loss
        total_loss = torch.mean(total_loss)
        return total_loss

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def accuracy_function(self, outputs:torch.Tensor, labels: torch.Tensor):
        _, predicted = torch.max(outputs, dim=-1)
        correct = (predicted == labels).sum()
        return correct/(labels.size(0)*labels.size(1))
        
    
    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor, batch_size: int, epochs: int = 1, reset_loss: int = 1):
        optimizer = optim.Adam(params=self.model.parameters())
        scheduler = ScheduledOptimizer(optimizer=optimizer, embedding_dim=self.embedding_dim, warmup_steps=4000)
        self.model = self.model.to(self.device)

        dataloader = self.build_dataset(x_train, y_train, batch_size)

        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0

            for index, data in enumerate(dataloader, 0):
                inputs, labels = data
                
                encoder_padding_mask = None
                decoder_padding_mask, look_ahead_mask = self.mask_generator.generate_mask(labels)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                decoder_padding_mask = decoder_padding_mask.to(self.device)
                look_ahead_mask = look_ahead_mask.to(self.device)

                scheduler.zero_grad()

                outputs = self.model(inputs, labels,encoder_padding_mask, decoder_padding_mask, look_ahead_mask, training=True)
                
                loss = self.loss_function(outputs, labels)

                loss.backward()
                scheduler.step()

                running_loss += loss.item()
                running_accuracy += self.accuracy_function(outputs, labels)
                if index%reset_loss*batch_size == 0:
                    print(f"Epoch: {epoch} Batch: {(index+1)} Accuracy: {(running_accuracy*100):.2f}% Loss: {(running_loss/(reset_loss*batch_size)):.3f}")
                    running_loss = 0.0
                    running_accuracy = 0.0
                

    

