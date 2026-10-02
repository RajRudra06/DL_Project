# nextSteps.md - read this first (for teammates and any AI assistant you use)

Project: Detection of Fake and AI-Generated Product Reviews Using Deep Learning (ICT 4442).
Task: binary text classification, label 1 = computer-generated (CG), label 0 = original human review (OR).
Dataset: Salminen et al. 2022 Fake Reviews Dataset, at `Project Files/DataSet/fake_reviews_dataset.csv`.
Code lives in `Project Files/code/`. The reports live in `Project Files/report/`.

## What already exists (Rudra's work, DO NOT EDIT these files)
- `src/evaluate.py` - common evaluation harness. Every model reports through it.
- `src/data.py` - loads the CSV, removes duplicate texts, makes ONE stratified 70/15/15 split (seed 42).
- `src/distilbert_model.py` - DistilBERT model (done, full run finished).
- `src/compare.py` - builds the comparison table from `results/*.json`.
- `results/distilbert*.json|csv` - DistilBERT results.
- `Project Files/report/` - draft report sections.
If you think one of these has a bug, message Rudra. Do not change it, because every model must run on the same split and the same metrics or the comparison is invalid.

## Ground rules
1. Use your own GitHub account. Commit small and often (model script, then run, then results), not one big push at the end. Commit history is graded for individual contribution.
2. Create your own new files only. One model script per owner, in `src/`.
3. Numbers must come from code you actually ran. No made-up, copied or estimated results. Fabricated results mean zero marks for that phase.
4. Any use of an AI tool for code or text must be disclosed in the report, with where it was used. You will be questioned on your own model in the viva, so read and understand every line you commit.
5. Run `git pull` before you start work every time.

## How to plug into the harness (required, copy this pattern)
```python
import time
import data                      # shared split
import evaluate as E             # shared metrics

train, val, test = data.load_splits()      # DataFrames with columns: text, label (plus category, rating)
# ... build and train your model on `train`, tune on `val` only ...
t0 = time.perf_counter()
# fit(...)
train_seconds = time.perf_counter() - t0

y_prob = ...  # numpy array: probability that each TEST review is class 1 (computer-generated)
latency = E.measure_latency(lambda xs: your_predict_proba(xs), test["text"].tolist()[:64], repeats=3)

E.evaluate_model(
    "Text-CNN",                  # EXACT name from the table below
    "Your Full Name",
    test["label"].to_numpy(), y_prob,
    n_params, train_seconds, latency_ms=latency,
    notes="epochs=..., device=cpu or cuda, key hyperparameters")
```
This writes `results/<model>.json`. Also save per-review test predictions to `results/<model>_test_predictions.csv` with columns `text,label,prob_cg` (see how `distilbert_model.py` does it, read it, do not edit it). Test data is used once, for the final numbers. Tune only on `val`.

Then run `python src/compare.py` to see your row in the table.

## Who builds what

| Owner | Model | Exact name string | Script to create | Priority |
|---|---|---|---|---|
| Vedant Agarwal (230953312) | TF-IDF + MLP | `TF-IDF + MLP` | `src/tfidf_mlp_model.py` | URGENT, needed for Interim |
| Ishan Satyanand Thakur (230953356) | Text-CNN | `Text-CNN` | `src/textcnn_model.py` | DONE |
| Vedant Totla (230911526) | BiLSTM + attention | `BiLSTM + Attention` | `src/bilstm_model.py` | after Interim, due 6 Oct |

Interim (due 2 Oct) needs at least two models with results. DistilBERT is one. The TF-IDF+MLP and the Text-CNN are the second and third, so these two have to run today.

### Vedant Agarwal - TF-IDF + MLP
- TF-IDF word n-grams (1-2) and character n-grams (3-5) from sklearn, fit on `train` only, then a small multilayer perceptron (for example PyTorch, 2 hidden layers, dropout) or sklearn `MLPClassifier`.
- Runs fine on a laptop CPU in minutes.
- Also yours (shared responsibility): dataset acquisition notes and an exploratory analysis script `src/eda.py` (class balance, review length, category and rating counts). The literature review table draft is in `Project Files/report/literature_review_draft.md`; read it and check it against the papers.

### Ishan Satyanand Thakur - Text-CNN (DONE)
- ~~Trainable word embeddings, parallel convolution filters of widths 3, 4, 5 with max-over-time pooling, dropout, a linear output layer (Kim, 2014).~~
- ~~Build the vocabulary from `train` only. Put tokenisation and embedding utilities in your own file `src/preprocess.py`. Do not change `data.py`.~~
- ~~Trains on CPU or on Colab with a GPU; say which in `notes`.~~

### Vedant Totla - BiLSTM + attention
- Bidirectional LSTM over word embeddings, attention pooling over the hidden states, linear output layer. Save the attention weights for a few test reviews, since they feed the error analysis.
- Later (not for Interim): cross-domain test on the Ott et al. hotel corpus (never trained on) and shared error analysis.

## What to send Rudra when done
1. Your pushed commits (he reads them from the repo history).
2. The path of your `results/<model>.json`.
3. Two or three sentences on what you built and what hyperparameters you used.
4. The list of tasks you completed, for the Individual Contribution Log. Each member signs that page by hand.

## Deadlines
- Interim report: 2 Oct 2026 (extended from 25 Sept), 11:59 PM via MS Forms.
- Final report, code and presentation: 31 Oct 2026. Presentations 26-29 Oct.
