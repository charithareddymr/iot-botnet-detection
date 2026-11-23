"""Load Bot-IoT data and turn it into model-ready features.

Leakage decisions come from the single-feature test in src/explore.py:
identifier-like columns (row IDs, IP addresses, ports, Argus sequence numbers)
are dropped, so models must learn from traffic BEHAVIOUR, not from details
of the lab network the dataset was recorded in.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN_CSV = DATA / "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
TEST_CSV = DATA / "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"

LABEL_COLUMNS = ["attack", "category", "subcategory"]
TARGET = "category"  # Normal, DDoS, DoS, Reconnaissance, Theft

# Dropped as leakage (see reports/data_summary.md):
#   pkSeqID           row ID: alone it separates normal from attack perfectly
#   saddr, daddr      IP addresses of the lab machines
#   sport, dport      ports specific to the lab setup
#   seq               Argus record sequence number (capture order, not behaviour)
IDENTIFIER_COLUMNS = ["pkSeqID", "saddr", "daddr", "sport", "dport", "seq"]

# Counted over the whole capture window. Kept, but tested separately in the
# ablation study because a real-time detector would not have whole-capture totals.
AGGREGATE_COLUMNS = ["N_IN_Conn_P_SrcIP", "N_IN_Conn_P_DstIP"]


def load(path):
    return pd.read_csv(path, low_memory=False)


def to_features(df, drop_aggregates=False):
    """Return (X, y). X has the behavioural features only; proto is one-hot encoded."""
    drop = IDENTIFIER_COLUMNS + LABEL_COLUMNS
    if drop_aggregates:
        drop = drop + AGGREGATE_COLUMNS
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = pd.get_dummies(X, columns=["proto"], prefix="proto", dtype=float)
    return X, df[TARGET].astype(str)


def align_columns(X_train, X_test):
    """Give the test set exactly the training columns (e.g. a protocol seen only in one split)."""
    return X_test.reindex(columns=X_train.columns, fill_value=0.0)


def rebalance(df, cap=50_000, floor=5_000, seed=42):
    """Training-only rebalancing: cap huge classes, oversample tiny ones (with replacement).

    Never applied to the test set, so evaluation reflects the real class mix.
    """
    parts = []
    for _, group in df.groupby(TARGET):
        if len(group) > cap:
            parts.append(group.sample(cap, random_state=seed))
        elif len(group) < floor:
            parts.append(group.sample(floor, replace=True, random_state=seed))
        else:
            parts.append(group)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
