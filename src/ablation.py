"""Leakage experiment: how much do identifier columns inflate results?

Retrains the best model (Random Forest) on four feature sets and evaluates
each on the same full, untouched test set:

  A. Behavioural features          the honest setup used in src/train.py
  B. Behavioural + identifiers     adds row ID, IP addresses, ports, sequence number
  C. Row ID only                   pkSeqID alone: knows nothing about network traffic
  D. Without connection counts     drops the two whole-capture aggregates,
                                   closer to what a real-time detector would see

Run from the project root:
    python -m src.ablation
Writes reports/ablation.md and reports/ablation.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.features import (
    IDENTIFIER_COLUMNS,
    LABEL_COLUMNS,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
    align_columns,
    load,
    rebalance,
    to_features,
)
from src.train import REPORTS, security_metrics

SEED = 42
LABELS = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"]


def encode_with_identifiers(train_df, test_df, columns=None):
    """Build features that INCLUDE identifier columns (for the leakage experiment only).

    Text columns (IP addresses, hex ports) become integer codes learned from the
    training data; values never seen in training get -1.
    """
    def base(df):
        X = df.drop(columns=LABEL_COLUMNS)
        return X[columns] if columns else X

    X_tr, X_te = base(train_df), base(test_df)
    for col in X_tr.columns:
        if col == "proto" or pd.api.types.is_numeric_dtype(X_tr[col]):
            continue
        categories = pd.Index(X_tr[col].astype(str).unique())
        X_tr[col] = pd.Categorical(X_tr[col].astype(str), categories=categories).codes
        X_te[col] = pd.Categorical(X_te[col].astype(str), categories=categories).codes
    if "proto" in X_tr.columns:
        X_tr = pd.get_dummies(X_tr, columns=["proto"], prefix="proto", dtype=float)
        X_te = pd.get_dummies(X_te, columns=["proto"], prefix="proto", dtype=float)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    return X_tr.fillna(-1), X_te.fillna(-1)


def run(name, X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_true = y_test.to_numpy()
    result = {
        "experiment": name,
        "n_features": X_train.shape[1],
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "normal_recall": float((y_pred[y_true == "Normal"] == "Normal").mean()),
        **security_metrics(y_true, y_pred),
    }
    # Which features did the model lean on most?
    top = np.argsort(model.feature_importances_)[::-1][:3]
    result["top_features"] = [
        f"{X_train.columns[i]} ({model.feature_importances_[i]:.2f})" for i in top
    ]
    print(f"{name}\n  balanced acc {result['balanced_accuracy']:.4f} | macro F1 {result['macro_f1']:.4f} | "
          f"false-positive rate {result['false_positive_rate']:.2%} | "
          f"top features: {', '.join(result['top_features'])}\n", flush=True)
    return result


def main():
    REPORTS.mkdir(exist_ok=True)
    print("Loading data...")
    train_raw, test_raw = load(TRAIN_CSV), load(TEST_CSV)
    train_bal = rebalance(train_raw)
    y_train, y_test = train_bal[TARGET].astype(str), test_raw[TARGET].astype(str)
    results = []

    X_tr, _ = to_features(train_bal)
    X_te, _ = to_features(test_raw)
    results.append(run("A. Behavioural features (honest)", X_tr, y_train, align_columns(X_tr, X_te), y_test))

    X_tr, X_te = encode_with_identifiers(train_bal, test_raw)
    results.append(run("B. Behavioural + identifiers (leaky)", X_tr, y_train, X_te, y_test))

    X_tr, X_te = encode_with_identifiers(train_bal, test_raw, columns=["pkSeqID"])
    results.append(run("C. Row ID only (pkSeqID)", X_tr, y_train, X_te, y_test))

    X_tr, _ = to_features(train_bal, drop_aggregates=True)
    X_te, _ = to_features(test_raw, drop_aggregates=True)
    results.append(run("D. Without whole-capture connection counts", X_tr, y_train, align_columns(X_tr, X_te), y_test))

    (REPORTS / "ablation.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Leakage experiment (Random Forest, full untouched test set)",
        "",
        f"Identifier columns: {', '.join(IDENTIFIER_COLUMNS)}",
        "",
        "| Experiment | Features | Balanced acc. | Macro F1 | Normal recall | False-positive rate | Top features (importance) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['experiment']} | {r['n_features']} | {r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} "
            f"| {r['normal_recall']:.3f} | {r['false_positive_rate']:.2%} | {'; '.join(r['top_features'])} |"
        )
    (REPORTS / "ablation.md").write_text("\n".join(lines) + "\n")
    print("Saved reports/ablation.md and reports/ablation.json")


if __name__ == "__main__":
    main()
