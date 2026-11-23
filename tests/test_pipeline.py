"""Tests for the data pipeline and evaluation logic.

They use small synthetic tables with the real Bot-IoT column names, so they run
in seconds without the 440 MB dataset (and in CI).

Run from the project root:
    python -m pytest -v
"""
import numpy as np
import pandas as pd
import pytest

from src.ablation import encode_with_identifiers
from src.explore import single_feature_leak_test
from src.features import (
    AGGREGATE_COLUMNS,
    IDENTIFIER_COLUMNS,
    LABEL_COLUMNS,
    align_columns,
    rebalance,
    to_features,
)
from src.train import security_metrics

CLASSES = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"]


def make_flows(counts, seed=0):
    """Synthetic flows. pkSeqID is assigned in class order, like a leaky row ID."""
    rng = np.random.default_rng(seed)
    category = np.concatenate([[c] * n for c, n in counts.items()])
    n = len(category)
    return pd.DataFrame({
        "pkSeqID": np.arange(n),
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "saddr": rng.choice(["192.168.100.3", "192.168.100.150"], n),
        "sport": rng.choice(["80", "0x0303", "1234"], n),
        "daddr": "192.168.100.5",
        "dport": rng.choice(["80", "443"], n),
        "seq": rng.integers(1, 100_000, n),
        "stddev": rng.random(n), "N_IN_Conn_P_SrcIP": rng.integers(1, 100, n),
        "min": rng.random(n), "state_number": rng.integers(1, 10, n), "mean": rng.random(n),
        "N_IN_Conn_P_DstIP": rng.integers(1, 100, n), "drate": rng.random(n),
        "srate": rng.random(n), "max": rng.random(n),
        "attack": (category != "Normal").astype(int),
        "category": category,
        "subcategory": "x",
    })


@pytest.fixture
def flows():
    return make_flows({"Normal": 40, "DDoS": 300, "DoS": 300, "Reconnaissance": 100, "Theft": 10})


def test_identifiers_and_labels_never_reach_the_model(flows):
    X, y = to_features(flows)
    for col in IDENTIFIER_COLUMNS + LABEL_COLUMNS:
        assert col not in X.columns
    assert set(y.unique()) == set(CLASSES)


def test_aggregates_kept_by_default_and_droppable(flows):
    assert set(AGGREGATE_COLUMNS) <= set(to_features(flows)[0].columns)
    X, _ = to_features(flows, drop_aggregates=True)
    assert not set(AGGREGATE_COLUMNS) & set(X.columns)


def test_protocol_is_one_hot_encoded(flows):
    X, _ = to_features(flows)
    assert "proto" not in X.columns
    assert {"proto_tcp", "proto_udp", "proto_icmp"} <= set(X.columns)


def test_test_set_gets_exactly_the_training_columns(flows):
    X_train, _ = to_features(flows)
    other = flows.copy()
    other["proto"] = "arp"  # protocol never seen in training
    X_test = align_columns(X_train, to_features(other)[0])
    assert list(X_test.columns) == list(X_train.columns)
    assert (X_test[["proto_tcp", "proto_udp", "proto_icmp"]] == 0).all().all()


def test_rebalance_caps_big_classes_and_oversamples_rare_ones(flows):
    balanced = rebalance(flows, cap=200, floor=50)
    counts = balanced["category"].value_counts()
    assert counts["DDoS"] == 200 and counts["DoS"] == 200          # capped
    assert counts["Normal"] == 50 and counts["Theft"] == 50        # oversampled
    assert counts["Reconnaissance"] == 100                          # untouched
    assert len(flows) == 750                                        # input not modified


def test_security_metrics():
    y_true = np.array(["Normal"] * 4 + ["DDoS"] * 6)
    y_pred = np.array(["Normal", "Normal", "Normal", "DoS"] + ["DDoS"] * 5 + ["Normal"])
    m = security_metrics(y_true, y_pred)
    assert m["false_positive_rate"] == pytest.approx(1 / 4)    # 1 normal flow flagged
    assert m["attack_detection_rate"] == pytest.approx(5 / 6)  # 1 attack missed
    assert m["normal_flows_in_test"] == 4


def test_identifier_values_unseen_in_training_become_minus_one(flows):
    test = flows.copy()
    test["saddr"] = "10.0.0.99"
    X_tr, X_te = encode_with_identifiers(flows, test)
    assert (X_te["saddr"] == -1).all()
    assert (X_tr["saddr"] >= 0).all()
    assert list(X_te.columns) == list(X_tr.columns)


def test_leak_test_flags_a_row_id_but_not_random_noise():
    flows = make_flows({"Normal": 200, "DDoS": 800}, seed=1)
    scores = dict(single_feature_leak_test(flows, max_rows=1000))
    assert scores["pkSeqID"] > 0.95   # row ID ordered by class: a perfect leak
    assert scores["mean"] < 0.7       # random noise: near chance
