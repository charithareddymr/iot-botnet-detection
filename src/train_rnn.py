"""Train the 5th model from the original project: a recurrent neural network (LSTM).

Same features, same training rebalancing and the same untouched test set as
src/train.py, so its results are directly comparable. The result is added to
reports/results.md and reports/results.json.

How the RNN sees the data: each network flow has 14 numeric features, which
are fed to the LSTM one at a time as a sequence of 14 steps. Flow features
are not truly sequential, so this mainly tests whether a recurrent model adds
anything over simpler models (a question worth asking, not assuming).

Run from the project root:
    python -m src.train_rnn
"""
import json
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.features import TEST_CSV, TRAIN_CSV, align_columns, load, rebalance, to_features
from src.train import REPORTS, evaluate, write_markdown

SEED = 42
LABELS = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"]
MODEL_NAME = "RNN (LSTM)"
EPOCHS = 15
PATIENCE = 3  # stop if validation macro F1 hasn't improved for this many epochs
BATCH_SIZE = 512


class FlowLSTM(nn.Module):
    def __init__(self, hidden_size=64, n_classes=len(LABELS)):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):  # x: (batch, n_features)
        out, _ = self.lstm(x.unsqueeze(-1))  # (batch, n_features, hidden)
        return self.head(out.mean(dim=1))  # mean-pool over the feature steps


def to_tensor(X, scaler):
    # Clip extreme values so a few huge packet rates can't destabilise training
    scaled = np.clip(scaler.transform(X.to_numpy(dtype=np.float32)), -10, 10)
    return torch.tensor(scaled, dtype=torch.float32)


@torch.no_grad()
def predict(model, X, batch_size=8192):
    model.eval()
    preds = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(preds).numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading data...")
    train_raw, test_raw = load(TRAIN_CSV), load(TEST_CSV)
    X, y = to_features(rebalance(train_raw))
    X_test, y_test = to_features(test_raw)
    X_test = align_columns(X, X_test)

    label_to_id = {label: i for i, label in enumerate(LABELS)}
    y_ids = y.map(label_to_id).to_numpy()

    # Hold out 10% of the (rebalanced) training data to decide when to stop
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_ids, test_size=0.1, stratify=y_ids, random_state=SEED
    )
    scaler = StandardScaler().fit(X_tr.to_numpy(dtype=np.float32))
    X_tr_t, X_val_t, X_test_t = (to_tensor(d, scaler) for d in (X_tr, X_val, X_test))
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    model = FlowLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training {MODEL_NAME} on {len(X_tr):,} rows, {X.shape[1]} features as a {X.shape[1]}-step sequence...")
    start = time.time()
    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X_tr_t))
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_f1 = f1_score(y_val, predict(model, X_val_t), average="macro")
        print(f"  epoch {epoch:2d}  loss {loss.item():.4f}  validation macro F1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print("  early stop: no improvement")
                break
    train_seconds = time.time() - start
    model.load_state_dict(best_state)

    y_pred = np.array(LABELS)[predict(model, X_test_t)]
    result = evaluate(MODEL_NAME, y_test.to_numpy(), y_pred, LABELS)
    result["train_seconds"] = train_seconds
    print(f"\n{MODEL_NAME}: balanced acc {result['balanced_accuracy']:.4f} | "
          f"macro F1 {result['macro_f1']:.4f} | "
          f"false-positive rate {result['false_positive_rate']:.2%} | "
          f"attack detection {result['attack_detection_rate']:.4%}")

    # Add to (or replace in) the results produced by src/train.py
    results_file = REPORTS / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results = [r for r in results if r["model"] != MODEL_NAME] + [result]
    results_file.write_text(json.dumps(results, indent=2))
    write_markdown(results, LABELS)
    print("Updated reports/results.md and reports/results.json")


if __name__ == "__main__":
    main()
