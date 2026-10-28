import time
import data
import evaluate as E

train, val, test = data.load_splits()
print(len(train), len(val), len(test))

print(train["label"].value_counts())
lengths = train["text"].str.split().str.len()
print(lengths.describe())
print(lengths.quantile([0.9, 0.95, 0.99]))

import re
from collections import Counter

def tokenize(text):
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9]", text.lower())

MAX_VOCAB = 30000
counter = Counter()
for t in train["text"]:
    counter.update(tokenize(t))

itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(MAX_VOCAB - 2)]
stoi = {w: i for i, w in enumerate(itos)}
print("vocab size:", len(itos))

import torch
from torch.utils.data import Dataset, DataLoader

MAX_LEN = 256

def encode(text):
    ids = [stoi.get(w, 1) for w in tokenize(text)][:MAX_LEN]   # 1 = <unk>
    ids += [0] * (MAX_LEN - len(ids))                           # 0 = <pad>
    return ids

class ReviewDS(Dataset):
    def __init__(self, df):
        self.x = torch.tensor([encode(t) for t in df["text"]], dtype=torch.long)
        self.y = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

train_dl = DataLoader(ReviewDS(train), batch_size=64, shuffle=True)
val_dl   = DataLoader(ReviewDS(val),   batch_size=256)
test_dl  = DataLoader(ReviewDS(test),  batch_size=256)

xb, yb = next(iter(train_dl))
print(xb.shape, yb.shape)

import torch.nn as nn

class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        mask = (x != 0)                              # [B, T] True for real tokens
        h, _ = self.lstm(self.drop(self.emb(x)))     # [B, T, 2H]
        scores = self.attn(torch.tanh(h)).squeeze(-1)   # [B, T]
        scores = scores.masked_fill(~mask, -1e9)        # ignore padding
        w = torch.softmax(scores, dim=1)                # attention weights
        ctx = (w.unsqueeze(-1) * h).sum(dim=1)          # [B, 2H]
        logit = self.out(self.drop(ctx)).squeeze(-1)    # [B]
        return logit, w


device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTMAttn(len(itos)).to(device)
logit, w = model(xb.to(device))
print(logit.shape, w.shape)
print("params:", sum(p.numel() for p in model.parameters()))