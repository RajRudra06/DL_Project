"""Text-CNN classifier (owner: Ishan Satyanand Thakur)."""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import data
import evaluate as E
import preprocess

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100, filter_sizes=(3, 4, 5), dropout_prob=0.5):
        super().__init__()
        self.embedding = preprocess.get_embedding_layer(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 2)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x) # (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1) # (batch_size, 1, seq_len, embed_dim)
        
        out = []
        for conv in self.convs:
            # conv(x): (batch_size, num_filters, seq_len - fs + 1, 1)
            c = F.relu(conv(x)).squeeze(3) # (batch_size, num_filters, seq_len - fs + 1)
            # max pool over time
            p = F.max_pool1d(c, c.size(2)).squeeze(2) # (batch_size, num_filters)
            out.append(p)
            
        out = torch.cat(out, 1) # (batch_size, len(filter_sizes) * num_filters)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


def predict_proba(model, tokenizer, texts, device, max_len, batch_size):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = [tokenizer.encode(t, max_len=max_len) for t in batch_texts]
            b = torch.tensor(encoded, dtype=torch.long, device=device)
            logits = model(b)
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=data.DEFAULT_CSV)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=100)
    ap.add_argument("--num-filters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=data.SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train, val, test = data.load_splits(args.csv, seed=args.seed)
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    tokenizer = preprocess.Tokenizer()
    tokenizer.fit(train["text"].tolist())
    
    model = TextCNN(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        filter_sizes=(3, 4, 5)
    )
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    texts = train["text"].tolist()
    labels = train["label"].to_numpy()
    
    encoded_train = [tokenizer.encode(t, max_len=args.max_len) for t in texts]
    train_x = torch.tensor(encoded_train, dtype=torch.long)
    train_y = torch.tensor(labels, dtype=torch.long)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    t_start = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        for i, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()
            opt.zero_grad()
            run_loss += loss.item()
            
        vp = predict_proba(model, tokenizer, val["text"].tolist(), device, args.max_len, args.batch_size)
        vm = E.compute_metrics(val["label"].to_numpy(), vp)
        print(f"epoch {ep + 1} loss {run_loss/len(train_loader):.4f} val acc {vm['accuracy']:.4f} macroF1 {vm['macro_f1']:.4f}")
        
    train_seconds = time.perf_counter() - t_start

    tp = predict_proba(model, tokenizer, test["text"].tolist(), device, args.max_len, args.batch_size)
    sample = test["text"].tolist()[:64]
    
    def inference_batch(xs):
        return predict_proba(model, tokenizer, xs, device, args.max_len, len(xs))
        
    latency = E.measure_latency(inference_batch, sample, repeats=3)

    res = E.evaluate_model(
        "Text-CNN", "Ishan Satyanand Thakur", test["label"].to_numpy(), tp,
        n_params, train_seconds, latency_ms=latency,
        notes=f"epochs={args.epochs}, device={device}, embed_dim={args.embed_dim}, num_filters={args.num_filters}, max_len={args.max_len}"
    )
    
    os.makedirs(E.RESULTS_DIR, exist_ok=True)
    pd.DataFrame({
        "text": test["text"], 
        "label": test["label"],
        "prob_cg": tp
    }).to_csv(
        os.path.join(E.RESULTS_DIR, "text-cnn_test_predictions.csv"), index=False
    )
    print(res["metrics"])

if __name__ == "__main__":
    main()
