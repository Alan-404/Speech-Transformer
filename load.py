#%%
import torch
from preprocessing.audio import AudioProcessor
from preprocessing.text import TextProcessor
import pandas as pd
import numpy as np
import librosa.display as display
import matplotlib.pyplot as plt
import pickle
path_dataset = "D:/Project/Dataset/speech-label"
# %%
df = pd.read_csv(f"{path_dataset}/meta_data.tsv", sep='\t')
# %%
df = df[:5000]
#%%
df.head(10)
# %%

# %%
sample_rate = 22050
duration = 10
frame_size = 551
hop_length = 220
mono = True

audio_handler = AudioProcessor(sample_rate=sample_rate, duration=duration, mono=mono, frame_size=frame_size, hop_length=hop_length, max=1)
# %%
files = np.array(df['file'])
# %%
X_train = []
for file in files:
    signal = audio_handler.process(f"{path_dataset}/{file}")
    X_train.append(signal)
#%%
X_train
# %%
display.specshow(X_train[0])
plt.colorbar()
# %%

# %%
X_train[0]
# %%
np.unique(X_train[0])
# %%

# %%
max_length = 40
# %%
text_handler = TextProcessor("./tokenizer")
# %%
# %%
sequences = np.array(df['label'])
# %%
sequences
# %%
y_train = text_handler.process(sequences, max_length=max_length)
# %%
y_train.shape
# %%
X_train = np.array(X_train)
# %%
X_train.shape
# %%
X_train = np.expand_dims(X_train, axis=1)
# %%
X_train.shape
# %%
with open('./clean/audio.pkl', 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
with open("./clean/text.pkl", 'wb') as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
y_train
# %%
np.max(y_train)
# %%
len(text_handler.tokenizer.word_index)
# %%
