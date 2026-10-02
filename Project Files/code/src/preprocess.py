import re
from collections import Counter
import torch
import torch.nn as nn

class Tokenizer:
    def __init__(self, max_vocab_size=20000, unk_token="<UNK>", pad_token="<PAD>"):
        self.max_vocab_size = max_vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.w2i = {pad_token: 0, unk_token: 1}
        self.i2w = {0: pad_token, 1: unk_token}
        self.vocab_size = 2

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        
        for w, _ in counts.most_common(self.max_vocab_size - 2):
            self.w2i[w] = self.vocab_size
            self.i2w[self.vocab_size] = w
            self.vocab_size += 1

    def encode(self, text, max_len=None):
        tokens = self.tokenize(text)
        indices = [self.w2i.get(w, self.w2i[self.unk_token]) for w in tokens]
        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices = indices + [self.w2i[self.pad_token]] * (max_len - len(indices))
        return indices

def get_embedding_layer(vocab_size, embed_dim=100):
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
