"""Fine-tuned DistilBERT classifier (owner: Rudra Rajpurohit).

Plain PyTorch training loop on top of Hugging Face DistilBERT, so every step can be
explained in the viva. Results are written through the common harness (evaluate.py).

Example (preliminary CPU run):
    python src/distilbert_model.py --train-size 4000 --eval-size 2000 --epochs 1
Full run:
    python src/distilbert_model.py --epochs 3
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import data
import evaluate as E


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len, padding=True,
               return_tensors="pt")


def predict_proba(model, tok, texts, device, max_len, batch_size):
    """Probability of class 1 (computer-generated) for each text."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = encode(tok, texts[i:i + batch_size], max_len)
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(**b).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=data.DEFAULT_CSV)
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--train-size", type=int, default=0, help="0 = full train split")
    ap.add_argument("--eval-size", type=int, default=0, help="0 = full val/test")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=data.SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print("device:", device)

    train, val, test = data.load_splits(args.csv, seed=args.seed)
    if args.train_size:
        train = train.sample(n=min(args.train_size, len(train)), random_state=args.seed)
    if args.eval_size:
        val = val.sample(n=min(args.eval_size, len(val)), random_state=args.seed)
        test = test.sample(n=min(args.eval_size, len(test)), random_state=args.seed)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                               num_labels=2)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(1, int(0.1 * steps)),
                           max(0.0, (steps - s) / max(1, steps * 0.9))))

    texts = train["text"].tolist()
    labels = train["label"].to_numpy()
    t_start = time.perf_counter()
    step = 0
    for ep in range(args.epochs):
        model.train()
        order = np.random.permutation(len(texts))
        run_loss = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            b = encode(tok, [texts[j] for j in idx], args.max_len)
            b = {k: v.to(device) for k, v in b.items()}
            y = torch.tensor(labels[idx], dtype=torch.long, device=device)
            loss = model(**b, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run_loss += loss.item()
            step += 1
            if step % 25 == 0:
                print(f"epoch {ep + 1} step {step}/{steps} loss {run_loss / 25:.4f}")
                run_loss = 0.0
        vp = predict_proba(model, tok, val["text"].tolist(), device, args.max_len,
                           args.batch_size)
        vm = E.compute_metrics(val["label"].to_numpy(), vp)
        print(f"epoch {ep + 1} val acc {vm['accuracy']:.4f} macroF1 {vm['macro_f1']:.4f}")
    train_seconds = time.perf_counter() - t_start

    tp = predict_proba(model, tok, test["text"].tolist(), device, args.max_len,
                       args.batch_size)
    sample = test["text"].tolist()[:64]
    latency = E.measure_latency(
        lambda xs: predict_proba(model, tok, xs, device, args.max_len, 16),
        sample, repeats=3)

    res = E.evaluate_model(
        "DistilBERT", "Rudra Rajpurohit", test["label"].to_numpy(), tp,
        n_params, train_seconds, latency_ms=latency,
        notes=(f"preliminary: train_size={len(train)}, epochs={args.epochs}, "
               f"max_len={args.max_len}, lr={args.lr}, device={device}"),
        extra={"val_metrics_last_epoch": vm, "args": vars(args)})
    os.makedirs(E.RESULTS_DIR, exist_ok=True)
    pd.DataFrame({"text": test["text"], "label": test["label"],
                  "prob_cg": tp}).to_csv(
        os.path.join(E.RESULTS_DIR, "distilbert_test_predictions.csv"), index=False)
    print(res["metrics"])


if __name__ == "__main__":
    main()
