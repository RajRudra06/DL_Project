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

import copy
import numpy as np
from sklearn.metrics import f1_score

torch.manual_seed(42)
crit = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def predict_probs(dl):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in dl:
            logit, _ = model(xb.to(device))
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)

EPOCHS, PATIENCE = 10, 2
best_f1, best_state, bad = 0.0, None, 0
y_val = val["label"].to_numpy()

t0 = time.perf_counter()
for ep in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logit, _ = model(xb)
        loss = crit(logit, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item() * len(yb)
    val_f1 = f1_score(y_val, predict_probs(val_dl) > 0.5)
    print(f"epoch {ep}  train_loss {total/len(train):.4f}  val_f1 {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
    else:
        bad += 1
        if bad >= PATIENCE:
            break
train_seconds = time.perf_counter() - t0
model.load_state_dict(best_state)
print("best val F1:", best_f1, "train seconds:", train_seconds)