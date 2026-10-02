"""Shared data loading and the common 70/15/15 split.

STAND-IN: the preprocessing pipeline is owned by Ishan (see README). This file only
provides what the evaluation harness and DistilBERT need: load the Salminen CSV,
drop exact-duplicate review texts, and make ONE stratified 70/15/15 split with a
fixed seed. Every model must use load_splits() so the split is identical.

Label convention: 1 = computer-generated (CG), 0 = original human review (OR).
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "..", "DataSet", "fake_reviews_dataset.csv")


def load_dataframe(path=DEFAULT_CSV):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def load_splits(path=DEFAULT_CSV, seed=SEED):
    """Stratified 70/15/15 split. Returns (train, val, test) DataFrames."""
    df = load_dataframe(path)
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"],
                                   random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"],
                                 random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


if __name__ == "__main__":
    tr, va, te = load_splits()
    for name, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"{name}: {len(d)} rows, label counts {d['label'].value_counts().to_dict()}")
