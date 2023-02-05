#%%
import torch
import pickle
from model.speech_transformer import SpeechTransformer
import numpy as np
# %%
sample_rate = 22050
duration = 8
frame_size = 551
hop_length = 220
mono = True
max_length = 40
# %%
with open("./clean/audio.pkl", 'rb') as handle:
    X_train = pickle.load(handle)
# %%
with open("./clean/text.pkl", 'rb') as handle:
    y_train = pickle.load(handle)
# %%
X_train.shape
# %%
y_train.shape
# %%
with open("./tokenizer/tokenizer.pkl", 'rb') as handle:
    tokenizer = pickle.load(handle)
# %%
vocab_size = len(tokenizer.word_index) + 1
# %%
vocab_size
# %%
model = SpeechTransformer(vocab_size=vocab_size, length_seq=max_length)
# %%
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
#%%
model.fit(x_train=X_train, y_train=y_train, batch_size=50, epochs=10)
# %%
model.save('./saved_models/speech_transformer.model')
# %%

# %%
type(y_train)
# %%

# %%
