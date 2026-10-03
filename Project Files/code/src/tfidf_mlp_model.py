"""TF-IDF + MLP classifier for Fake Review Detection.

Owner: Vedant Agarwal (230953312)

Architecture:
- Feature Extraction:
  - TF-IDF word n-grams (1, 2) [sublinear TF, min_df=2, max_features=15,000]
  - TF-IDF character n-grams (3, 5) [sublinear TF, min_df=5, max_features=25,000]
  - FeatureUnion combined (total 40,000 sparse input features)
  - Fitted STRICTLY on train split only
- Classifier:
  - Multilayer Perceptron (MLP) with 2 hidden layers: (128, 64)
  - ReLU activation, Adam optimizer, L2 regularization (alpha=1e-4)
  - Tuned on val split, test split evaluated strictly once at the end
- Evaluation:
  - Uses shared harness (evaluate.py) reporting accuracy, precision, recall,
    macro F1, ROC-AUC, parameter count, training time, and inference latency.
  - Generates results/tfidf_mlp.json and results/tfidf_mlp_test_predictions.csv.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion

# Add src to path for imports
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data
import evaluate as E


def build_feature_extractor(word_features=15000, char_features=25000):
    """Build FeatureUnion with word (1, 2) and character (3, 5) n-grams."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=word_features,
        sublinear_tf=True,
        min_df=2
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char",
        max_features=char_features,
        sublinear_tf=True,
        min_df=5
    )
    return FeatureUnion([
        ("word", word_vec),
        ("char", char_vec)
    ])


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + MLP Fake Review Classifier")
    parser.add_argument("--csv", default=data.DEFAULT_CSV, help="Path to fake_reviews_dataset.csv")
    parser.add_argument("--word-features", type=int, default=15000, help="Max word n-gram features")
    parser.add_argument("--char-features", type=int, default=25000, help="Max char n-gram features")
    parser.add_argument("--hidden-dim1", type=int, default=128, help="Hidden layer 1 size")
    parser.add_argument("--hidden-dim2", type=int, default=64, help="Hidden layer 2 size")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization weight")
    parser.add_argument("--seed", type=int, default=data.SEED, help="Random seed")
    args = parser.parse_args()

    print("=" * 65)
    print("MODEL: TF-IDF + MLP")
    print("Author: Vedant Agarwal (230953312)")
    print(f"Configuration: word_ngrams=(1,2) [{args.word_features:,}], char_ngrams=(3,5) [{args.char_features:,}]")
    print(f"MLP Hidden Architecture: ({args.hidden_dim1}, {args.hidden_dim2}) | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 65)

    # 1. Load shared 70/15/15 stratified splits
    train, val, test = data.load_splits(args.csv, seed=args.seed)
    print(f"Dataset splits: train={len(train):,} | val={len(val):,} | test={len(test):,}")

    # 2. Fit TF-IDF feature extraction strictly on train only
    print("\n[Step 1/4] Fitting TF-IDF vectorizers on train split only...")
    feature_extractor = build_feature_extractor(
        word_features=args.word_features,
        char_features=args.char_features
    )
    t_feat_start = time.perf_counter()
    X_train = feature_extractor.fit_transform(train["text"])
    feat_fit_seconds = time.perf_counter() - t_feat_start
    print(f"  Feature extraction fitted in {feat_fit_seconds:.2f}s | Vocabulary size: {X_train.shape[1]:,} features")

    # Transform val and test splits
    X_val = feature_extractor.transform(val["text"])
    X_test = feature_extractor.transform(test["text"])

    y_train = train["label"].to_numpy()
    y_val = val["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # 3. Initialize and train MLP
    print("\n[Step 2/4] Training MLP classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(args.hidden_dim1, args.hidden_dim2),
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.lr,
        max_iter=args.epochs,
        random_state=args.seed,
        verbose=True
    )

    t_train_start = time.perf_counter()
    mlp.fit(X_train, y_train)
    mlp_train_seconds = time.perf_counter() - t_train_start
    total_train_seconds = feat_fit_seconds + mlp_train_seconds
    print(f"  MLP training completed in {mlp_train_seconds:.2f}s (Total train time incl. vectorizer: {total_train_seconds:.2f}s)")

    # 4. Tune / Validate on val split
    print("\n[Step 3/4] Evaluating on validation split (tuning)...")
    val_probs = mlp.predict_proba(X_val)[:, 1]
    val_metrics = E.compute_metrics(y_val, val_probs)
    print(f"  Val Accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  Val Precision: {val_metrics['precision']:.4f}")
    print(f"  Val Recall   : {val_metrics['recall']:.4f}")
    print(f"  Val Macro-F1 : {val_metrics['macro_f1']:.4f}")
    print(f"  Val ROC-AUC  : {val_metrics['roc_auc']:.4f}")

    # 5. Final evaluation on test split (used strictly once)
    print("\n[Step 4/4] Final evaluation on test split...")
    test_probs = mlp.predict_proba(X_test)[:, 1]

    # Parameter count calculation (weights + biases across all layers)
    n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    print(f"  Total trainable parameters: {n_params:,}")

    # Inference latency measurement (mean ms per review on 64 test samples, 3 repeats)
    sample_texts = test["text"].tolist()[:64]
    def predict_batch(texts):
        return mlp.predict_proba(feature_extractor.transform(texts))[:, 1]

    latency_ms = E.measure_latency(predict_batch, sample_texts, repeats=3)
    print(f"  Inference Latency: {latency_ms:.4f} ms/review")

    # Report through shared harness
    notes = (f"epochs={args.epochs}, device=cpu, word_features={args.word_features}, "
             f"char_features={args.char_features}, hidden=({args.hidden_dim1},{args.hidden_dim2}), "
             f"batch_size={args.batch_size}")

    res = E.evaluate_model(
        "TF-IDF + MLP",
        "Vedant Agarwal",
        y_test,
        test_probs,
        n_params,
        total_train_seconds,
        latency_ms=latency_ms,
        notes=notes,
        extra={
            "val_metrics": val_metrics,
            "feature_fit_seconds": feat_fit_seconds,
            "mlp_train_seconds": mlp_train_seconds,
            "args": vars(args)
        },
        save=True
    )

    # Ensure results/tfidf_mlp.json is explicitly generated
    os.makedirs(E.RESULTS_DIR, exist_ok=True)
    explicit_json_path = os.path.join(E.RESULTS_DIR, "tfidf_mlp.json")
    with open(explicit_json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved results JSON to: {explicit_json_path}")

    # Save per-review test predictions CSV
    predictions_csv = os.path.join(E.RESULTS_DIR, "tfidf_mlp_test_predictions.csv")
    pd.DataFrame({
        "text": test["text"],
        "label": test["label"],
        "prob_cg": test_probs
    }).to_csv(predictions_csv, index=False)
    print(f"Saved test predictions CSV to: {predictions_csv}")

    print("\n" + "=" * 65)
    print("FINAL TEST METRICS:")
    m = res["metrics"]
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  Macro-F1  : {m['macro_f1']:.4f}")
    print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"  Confusion : {m['confusion']}")
    print(f"  Train Time: {total_train_seconds:.1f}s | Latency: {latency_ms:.2f} ms/review")
    print("=" * 65)


if __name__ == "__main__":
    main()
