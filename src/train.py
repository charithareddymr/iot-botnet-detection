"""Train Random Forest, AdaBoost, KNN and MLP on Bot-IoT and evaluate them honestly.

Evaluation uses the FULL, untouched test set and reports metrics that are not
fooled by class imbalance:
  - per-class recall, precision and F1
  - macro F1 and balanced accuracy (every class counts equally)
  - false-positive rate: share of NORMAL traffic wrongly flagged as an attack
  - attack detection rate: share of attacks flagged as any attack

Run from the project root:
    python -m src.train
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import (
    ROOT,
    TEST_CSV,
    TRAIN_CSV,
    align_columns,
    load,
    rebalance,
    to_features,
)

REPORTS = ROOT / "reports"
SEED = 42


def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
            n_estimators=200,
            random_state=SEED,
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=100, early_stopping=True, random_state=SEED
            ),
        ),
    }


def security_metrics(y_true, y_pred):
    """Binary view a security team cares about: normal vs any attack."""
    is_normal = y_true == "Normal"
    flagged = y_pred != "Normal"
    return {
        "false_positive_rate": float(flagged[is_normal].mean()),
        "attack_detection_rate": float(flagged[~is_normal].mean()),
        "normal_flows_in_test": int(is_normal.sum()),
    }


def evaluate(name, y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    result = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in labels},
        **security_metrics(y_true, y_pred),
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize="true", values_format=".2f",
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"{name}: share of each true class (rows)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(REPORTS / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
    return result


def write_markdown(results, labels):
    lines = [
        "# Results (identifier columns removed, full untouched test set)",
        "",
        "| Model | Accuracy | Balanced acc. | Macro F1 | False-positive rate | Attack detection | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} "
            f"| {r['false_positive_rate']:.2%} | {r['attack_detection_rate']:.4%} | {r['train_seconds']:.0f}s |"
        )
    lines += ["", "## Recall per class (share of each class correctly identified)", "",
              "| Model | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for r in results:
        lines.append(f"| {r['model']} | " + " | ".join(f"{r['per_class'][c]['recall']:.3f}" for c in labels) + " |")
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n")


def main():
    REPORTS.mkdir(exist_ok=True)
    print("Loading data...")
    train_raw, test_raw = load(TRAIN_CSV), load(TEST_CSV)

    train_bal = rebalance(train_raw)
    X_train, y_train = to_features(train_bal)
    X_test, y_test = to_features(test_raw)
    X_test = align_columns(X_train, X_test)
    labels = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"]

    print(f"Training rows after rebalancing: {len(X_train):,}")
    print(y_train.value_counts().to_string())
    print(f"Test rows (untouched): {len(X_test):,}\n")
    print(f"Features used ({X_train.shape[1]}): {', '.join(X_train.columns)}\n")

    results = []
    for name, model in build_models().items():
        print(f"Training {name}...", flush=True)
        start = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - start
        y_pred = model.predict(X_test)
        r = evaluate(name, y_test.to_numpy(), y_pred, labels)
        r["train_seconds"] = train_seconds
        results.append(r)
        print(f"  balanced acc {r['balanced_accuracy']:.4f} | macro F1 {r['macro_f1']:.4f} | "
              f"false-positive rate {r['false_positive_rate']:.2%} | "
              f"attack detection {r['attack_detection_rate']:.4%}\n")

    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, labels)
    print("Saved reports/results.md, reports/results.json and confusion matrix images.")


if __name__ == "__main__":
    main()
