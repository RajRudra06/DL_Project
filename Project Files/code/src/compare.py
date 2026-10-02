"""Builds the single comparison table across all models (owner: Rudra Rajpurohit).

Reads every results/<model>.json written by evaluate.evaluate_model() and writes
results/comparison_table.md and results/comparison_table.csv. Models that have not
been run yet appear as "pending" so the table always shows all four rows.

    python src/compare.py
"""
import csv
import glob
import json
import os

import evaluate as E

ORDER = ["TF-IDF + MLP", "Text-CNN", "BiLSTM + Attention", "DistilBERT"]
KEYS = {  # tolerate the different spellings the harness can produce
    "tfidf_mlp": "TF-IDF + MLP", "tf-idf_mlp": "TF-IDF + MLP",
    "text-cnn": "Text-CNN", "textcnn": "Text-CNN",
    "bilstm_attention": "BiLSTM + Attention", "bilstm__attention": "BiLSTM + Attention",
    "distilbert": "DistilBERT",
}
COLS = ["Model", "Owner", "Acc", "Prec", "Rec", "Macro-F1", "ROC-AUC",
        "Params", "Train (s)", "Latency (ms/review)", "Notes"]


def canonical(res, fname):
    stem = os.path.splitext(os.path.basename(fname))[0].lower()
    return KEYS.get(stem, res.get("model", stem))


def row(res):
    m = res["metrics"]
    lat = res.get("latency_ms_per_review")
    return [res["model"], res["owner"], f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
            f"{m['recall']:.4f}", f"{m['macro_f1']:.4f}", f"{m['roc_auc']:.4f}",
            f"{res['n_params']:,}", f"{res['train_seconds']:.0f}",
            "-" if lat is None else f"{lat:.2f}", res.get("notes", "")]


def main():
    found = {}
    for f in glob.glob(os.path.join(E.RESULTS_DIR, "*.json")):
        res = json.load(open(f))
        if "metrics" in res:
            found[canonical(res, f)] = res
    rows = []
    for name in ORDER + [k for k in found if k not in ORDER]:
        rows.append(row(found[name]) if name in found
                    else [name, "-"] + ["pending"] * 8 + [""])
    out_md = os.path.join(E.RESULTS_DIR, "comparison_table.md")
    with open(out_md, "w") as f:
        f.write("| " + " | ".join(COLS) + " |\n")
        f.write("|" + "---|" * len(COLS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    with open(os.path.join(E.RESULTS_DIR, "comparison_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    print(open(out_md).read())


if __name__ == "__main__":
    main()
