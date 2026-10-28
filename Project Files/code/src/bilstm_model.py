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