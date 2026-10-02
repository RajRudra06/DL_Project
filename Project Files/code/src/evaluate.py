"""Common evaluation harness shared by all four models.

Every model reports through evaluate_model() so that metrics are computed the
same way: accuracy, precision, recall, macro F1, ROC-AUC, parameter count,
training time and inference latency.

Convention: label 1 = computer-generated (CG, the "fake" class), 0 = original (OR).
"""
import json
import os
import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def compute_metrics(y_true, y_prob, threshold=0.5):
    """y_prob = predicted probability of class 1 (computer-generated)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def measure_latency(predict_fn, sample, repeats=5):
    """Mean milliseconds per review for predict_fn(list_of_inputs)."""
    predict_fn(sample[:8])  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(sample)
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.mean(times)) / len(sample)


def evaluate_model(name, owner, y_true, y_prob, n_params, train_seconds,
                   latency_ms=None, notes="", split="test", extra=None,
                   save=True):
    """Compute metrics and write results/<name>.json. Returns the result dict."""
    res = {
        "model": name,
        "owner": owner,
        "split": split,
        "n_eval": int(len(y_true)),
        "metrics": compute_metrics(y_true, y_prob),
        "n_params": int(n_params),
        "train_seconds": float(train_seconds),
        "latency_ms_per_review": None if latency_ms is None else float(latency_ms),
        "notes": notes,
    }
    if extra:
        res["extra"] = extra
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fname = name.lower().replace(" ", "_").replace("+", "").replace("/", "_")
        with open(os.path.join(RESULTS_DIR, fname + ".json"), "w") as f:
            json.dump(res, f, indent=2)
    return res
