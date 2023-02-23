import torch
from argparse import ArgumentParser
from model.speech_transformer import SpeechTransformer
from preprocessing.text import TextProcessor
from typing import Union, Callable
parser = ArgumentParser()

parser.add_argument('--sample_rate', type=int, default=22050)
parser.add_argument('--duration', type=float, default=10.0)
parser.add_argument("--frame_size", type=int, default=550)
parser.add_argument("--hop_length", type=int, default=220)
parser.add_argument("--n_e", type=int, default=12)
parser.add_argument('--n_d', type=int, default=6)
parser.add_argument("--embedding_dim", type=int, default=256)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--d_ff", type=int, default=2048)
parser.add_argument("--dropout_rate", type=float, default=0.1)
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--activation", default=torch.nn.functional.relu)
parser.add_argument("--m", type=int, default=2)
parser.add_argument("--channels", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--mini_batch", type=int, default=1)

parser.add_argument("--tokenizer", type=str)
parser.add_argument("--audio", type=str)
parser.add_argument("--text", type=str)
parser.add_argument("--checkpoint", type=str)

args = parser.parse_args()


def program(audio_path: str, 
            tokenizer_path: str,
            text_path: str, 
            sample_rate: int, 
            duration: float, 
            frame_size: int, 
            hop_length: int, 
            n_e: int, 
            n_d: int, 
            embedding_dim: int, 
            heads: int, 
            d_ff: int, 
            dropout_rate: float, 
            eps:float, 
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]],
            m: int,
            channels: int,
            checkpoint: str,
            batch_size: int,
            epochs: int,
            mini_batch: int):
        text_processor = TextProcessor(tokenizer_path=tokenizer_path)

        audio_data = text_processor.load_data(audio_path)
        text_data = text_processor.load_data(text_path)

        audio_data = torch.tensor(audio_data)
        text_data = torch.tensor(text_data)

        text_processor.loadd_tokenizer(tokenizer_path)

        vocab_size = text_processor.tokenizer.num_tokens + 1
        length_seq = text_data.size(1) - 1

        model = SpeechTransformer(vocab_size=vocab_size, sample_rate=sample_rate, duration=duration, frame_size=frame_size, hop_length=hop_length, length_seq=length_seq, n_e=n_e, n_d=n_d, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation, m=m, channels=channels, checkpoint=checkpoint)

        model.fit(audio_data, text_data, batch_size=batch_size, epochs=epochs, reset_loss=mini_batch)

if __name__ == "__main__":
    if args.tokenizer is None or args.audio is None or args.text is None or args.checkpoint is None:
        print("Missing Information")
    else:
        program(
            audio_path=args.audio,
            tokenizer_path=args.tokenizer,
            text_path=args.text,
            sample_rate=args.sample_rate,
            duration=args.duration,
            frame_size=args.frame_size,
            hop_length=args.hop_length,
            n_e=args.n_e,
            n_d=args.n_d,
            embedding_dim=args.embedding_dim,
            heads=args.heads,
            d_ff=args.d_ff,
            dropout_rate=args.dropout_rate,
            eps=args.eps,
            activation=args.activation,
            m=args.m,
            channels=args.channels,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            epochs=args.epochs,
            mini_batch=args.mini_batch
        )


