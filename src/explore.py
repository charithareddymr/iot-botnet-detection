"""Explore the Bot-IoT 5% (10-best features) training and testing data.

Answers two questions before any model is trained:
  1. What is in the data, and how imbalanced are the classes?
  2. Can a model "cheat"? Each column is tested on its own: if a single column
     predicts attack vs normal almost perfectly, it is probably an identifier
     or artifact of how the dataset was recorded (leakage), not real behaviour.

Run from the project root:
    python -m src.explore
Writes a summary to reports/data_summary.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
TRAIN_CSV = DATA / "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
TEST_CSV = DATA / "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"

LABEL_COLUMNS = ["attack", "category", "subcategory"]


def load(path):
    # Ports can be written in hex (e.g. "0x0303"), so read everything that
    # isn't clearly numeric as text and let pandas sort it out.
    return pd.read_csv(path, low_memory=False)


def as_numeric(series):
    """Convert a column to numbers so a tree can split on it.
    Text values (IP addresses, protocols, hex ports) become category codes."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(-1)
    return series.astype(str).astype("category").cat.codes


def single_feature_leak_test(df, label="attack", max_rows=300_000, seed=42):
    """Train a tiny decision tree on ONE column at a time and score it.

    Keeps every normal row (they are rare) plus a random sample of attacks,
    then measures balanced accuracy, which is not fooled by class imbalance.
    """
    normal = df[df[label] == 0]
    attacks = df[df[label] == 1]
    n_attacks = min(len(attacks), max_rows - len(normal))
    sample = pd.concat([normal, attacks.sample(n_attacks, random_state=seed)])
    y = sample[label].to_numpy()

    results = []
    for col in sample.columns:
        if col in LABEL_COLUMNS:
            continue
        x = as_numeric(sample[col]).to_numpy().reshape(-1, 1)
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=seed
        )
        tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)
        tree.fit(x_tr, y_tr)
        results.append((col, balanced_accuracy_score(y_te, tree.predict(x_te))))
    return sorted(results, key=lambda r: r[1], reverse=True)


def counts_table(series):
    counts = series.value_counts()
    pct = (counts / counts.sum() * 100).round(4)
    return pd.DataFrame({"rows": counts, "percent": pct})


def main():
    REPORTS.mkdir(exist_ok=True)
    print("Loading data (this takes a minute)...")
    train, test = load(TRAIN_CSV), load(TEST_CSV)
    lines = ["# Bot-IoT data summary", ""]

    def section(title, body):
        print(f"\n=== {title} ===\n{body}")
        lines.extend([f"## {title}", "", "```", str(body), "```", ""])

    section("Shape", f"train: {train.shape[0]:,} rows x {train.shape[1]} columns\n"
                     f"test:  {test.shape[0]:,} rows x {test.shape[1]} columns")
    section("Columns and types", train.dtypes.to_string())
    section("Missing values (train)", train.isna().sum().loc[lambda s: s > 0].to_string() or "none")

    for name, df in [("train", train), ("test", test)]:
        section(f"attack label ({name}): 0 = normal, 1 = attack", counts_table(df["attack"]).to_string())
    section("category (train)", counts_table(train["category"]).to_string())
    section("subcategory (train)", counts_table(train["subcategory"]).to_string())

    always_attack = (test["attack"] == 1).mean()
    section(
        "The accuracy trap",
        f"A 'model' that always answers ATTACK scores {always_attack:.4%} accuracy on the test set,\n"
        f"while catching 0 of {int((test['attack'] == 0).sum())} normal flows.\n"
        "So accuracy is meaningless here: we must report per-class recall,\n"
        "balanced accuracy, and false-positive rate instead.",
    )

    print("\nRunning single-feature leak test (about a minute)...")
    leak = single_feature_leak_test(train)
    leak_text = "\n".join(
        f"{col:22s} {score:.3f}" + ("   <-- suspicious" if score >= 0.95 else "")
        for col, score in leak
    )
    section(
        "Single-feature leak test (balanced accuracy using ONE column alone; 0.5 = random)",
        leak_text,
    )

    out = REPORTS / "data_summary.md"
    out.write_text("\n".join(lines))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
