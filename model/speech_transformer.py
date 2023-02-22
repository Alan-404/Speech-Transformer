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

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class SpeechTransformerModel(nn.Module):
    def __init__(self, vocab_size: int, 
                sample_rate: int,
                duration: float,
                frame_size: int,
                hop_length: int,
                length_seq: int, 
                n_e: int, 
                n_d: int, 
                embedding_dim: int, 
                heads: int, 
                d_ff: int, 
                dropout_rate: float, 
                eps: float, 
                activation: Union[str, Callable[[torch.Tensor], torch.Tensor]],
                m: int,
                channels: int):
        super().__init__()
        self.encoder = Encoder(n=n_e, length=length_seq, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation, m=m, channels=channels, sample_rate=sample_rate, duration=duration, frame_size=frame_size, hop_length=hop_length)
        self.decoder = Decoder(vocab_size=vocab_size, n=n_d, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)

        self.mask_generator = MaskGenerator()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, encoder_padding_mask: torch.Tensor,decoder_padding_mask: torch.Tensor, look_ahead_mask: torch.Tensor, training: bool):
        
        encoder_output = self.encoder(inputs, encoder_padding_mask, training)
        decoder_output = self.decoder(labels, encoder_output, look_ahead_mask, decoder_padding_mask, training)

        return decoder_output

class SpeechTransformer:
    def __init__(self, 
                vocab_size: int, 
                sample_rate: int,
                duration: float,
                frame_size: int,
                hop_length: int,
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
                checkpoint: str = None):
        self.model = SpeechTransformerModel(
            vocab_size=vocab_size,
            sample_rate=sample_rate,
            duration=duration,
            frame_size=frame_size, 
            hop_length=hop_length, length_seq=length_seq,
            n_e=n_e,
            n_d=n_d,
            embedding_dim=embedding_dim,
            heads=heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            eps=eps,
            activation=activation,
            m=m,
            channels=channels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.mask_generator = MaskGenerator()
        self.checkpoint = checkpoint

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
        if self.checkpoint is not None:
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
                targets = labels[:, 1:]
                labels = labels[:, :-1]
                
                encoder_padding_mask = None
                decoder_padding_mask, look_ahead_mask = self.mask_generator.generate_mask(labels)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                targets = targets.to(self.device)
                decoder_padding_mask = decoder_padding_mask.to(self.device)
                look_ahead_mask = look_ahead_mask.to(self.device)

                scheduler.zero_grad()

                outputs = self.model(inputs, labels,encoder_padding_mask, decoder_padding_mask, look_ahead_mask, training=True)
                
                loss = self.loss_function(outputs, targets)

                loss.backward()
                scheduler.step()

                running_loss += loss.item()
                running_accuracy += self.accuracy_function(outputs, labels)
                if index%reset_loss*batch_size == 0:
                    print(f"Epoch: {epoch} Batch: {(index+1)} Accuracy: {(running_accuracy*100):.2f}% Loss: {(running_loss/(reset_loss*batch_size)):.3f}")
                    running_loss = 0.0
                    running_accuracy = 0.0

    def predict(self, encoder_in: torch.Tensor, decoder_in: torch.Tensor, max_len: int, end_token: int):
        self.load(self.checkpoint)
        for _ in range(max_len):
            encoder_padding_mask = None
            decoder_padding_mask, look_ahead_mask = self.mask_generator.generate_mask(decoder_in)

            output = self.model(encoder_in, decoder_in, encoder_padding_mask, look_ahead_mask, decoder_padding_mask, False)

            output = output[:, -1, :]

            _, predicted = torch.max(output)

            if predicted == end_token:
                break

            result = torch.concat([decoder_in, predicted])
        return result

    

