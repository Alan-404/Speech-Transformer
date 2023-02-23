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
                duration: int,
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
                duration: int,
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

        self.criterion = nn.CrossEntropyLoss()

        self.embedding_dim = embedding_dim
        self.mask_generator = MaskGenerator()
        self.checkpoint = checkpoint

        self.epoch = 0
        self.optimizer = optim.Adam(params=self.model.parameters())
        self.scheduler = ScheduledOptimizer(optimizer=self.optimizer, embedding_dim=self.embedding_dim, warmup_steps=4000)
        self.model = self.model.to(device)

    def build_dataset(self, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int):
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def loss_function(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = labels.size(1)
        loss = 0.0
        for index in range(batch_size):
            loss += self.criterion(outputs[index], labels[index])
        loss = loss/batch_size
        return loss

    def __save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch
        }, path)

    def save_model(self, path: str = None):
        if path is None and self.checkpoint is not None:
            self.__save_model(self.checkpoint)
        elif path is not None:
            self.__save_model(path)
            self.checkpoint = path
    def __load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scheduler.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.model.eval()
        
    def load(self, path: str = None):
        if path is None and self.checkpoint is not None:
            self.__load_model(self.checkpoint)
        elif path is not None:
            self.__load_model(path)
            self.checkpoint = path

    def accuracy_function(self, outputs:torch.Tensor, labels: torch.Tensor):
        _, predicted = torch.max(outputs, dim=-1)
        correct = (predicted == labels).sum()
        return correct/(labels.size(0)*labels.size(1))
        
    
    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor, batch_size: int, epochs: int = 1, reset_loss: int = 1):
        

        dataloader = self.build_dataset(x_train, y_train, batch_size)

        for _ in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0

            for index, data in enumerate(dataloader, 0):
                inputs, labels = data
                targets = labels[:, 1:]
                labels = labels[:, :-1]
                
                encoder_padding_mask = None
                decoder_padding_mask, look_ahead_mask = self.mask_generator.generate_mask(labels)

                inputs = inputs.to(device)
                labels = labels.to(device)
                targets = targets.to(device)
                decoder_padding_mask = decoder_padding_mask.to(device)
                look_ahead_mask = look_ahead_mask.to(device)

                self.scheduler.zero_grad()

                outputs = self.model(inputs, labels,encoder_padding_mask, decoder_padding_mask, look_ahead_mask, training=True)
                
                loss = self.loss_function(outputs, targets)

                loss.backward()
                self.scheduler.step()

                running_loss += loss.item()
                running_accuracy += self.accuracy_function(outputs, labels)
                if index%reset_loss == 0:
                    print(f"Epoch: {self.epoch+1} Batch: {(index+1)} Accuracy: {(running_accuracy*100):.2f}% Loss: {(running_loss/(reset_loss)):.3f}")
                    running_loss = 0.0
                    running_accuracy = 0.0
            
            self.epoch += 1

        if self.checkpoint is not None:
            self.save_model(self.checkpoint)

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

    

