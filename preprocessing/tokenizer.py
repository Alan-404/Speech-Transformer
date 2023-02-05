import numpy as np
import re
class Tokenizer:
    def __init__(self):
        self.word_index = dict()
        self.word_counts = dict()
        self.count = 0

    def processing_sequence(self, sequence: str):
        sequence = sequence.lower()
        re.sub(r'([?.!,¿@=+/#])', '', sequence)
        re.sub(r'\s\s+', ' ', sequence)
        sequence = sequence.strip()
        sequence = f"<start> {sequence} <end>"
        return sequence
    def fit_to_tokenizer(self, sequence: str):
        sequence = self.processing_sequence(sequence)
        words = sequence.split(' ')
        for word in words:
            if word not in self.word_index:
                self.count += 1
                self.word_index[word] = self.count
                self.word_counts[word] = 1
            else:
                self.word_counts[word] += 1

    def fit_number(self, sequence: str):
        words = sequence.split(' ')
        arr = np.array([], dtype=np.int64)
        for word in words:
            arr = np.append(arr, self.word_index[word])
        return arr

    def fit_to_texts(self, sequences):
        for sequence in sequences:
            self.fit_to_tokenizer(sequence)

    def texts_to_sequences(self, sequences: list):
        result = []
        for sequence in sequences:
            sequence = self.processing_sequence(sequence)
            sequence = self.fit_number(sequence)
            result.append(sequence)

        return result

    def padding_sequence(self, sequence, padding: str, maxlen: int):
        delta = maxlen - len(sequence)
        zeros = np.zeros(delta, dtype=np.int64)

        if padding.strip().lower() == 'post':
            return np.concatenate((sequence, zeros), axis=0)
        elif padding.strip().lower() == 'pre':
            return np.concatenate((zeros, sequence), axis=0)

    def truncating_sequence(self, sequence, truncating: str, maxlen: int):
        if truncating.strip().lower() == 'post':
            return sequence[0:maxlen]
        elif truncating.strip().lower() == 'pre':
            delta = sequence.shape[0] - maxlen
            return sequence[delta: len(sequence)]

    def pad_sequences(self, sequences: list, maxlen: int, padding: str = 'post', truncating: str = 'post'):
        result = []
        for index, sequence in enumerate(sequences):
            delta = sequence.shape[0] - maxlen
            if delta < 0:
                sequence = self.padding_sequence(sequence, padding, maxlen)
            elif delta > 0:
                sequence = self.truncating_sequence(sequence, truncating, maxlen)
            result.append(sequence)
        
        return np.array(result)