# Leakage experiment (Random Forest, full untouched test set)

Identifier columns: pkSeqID, saddr, daddr, sport, dport, seq

| Experiment | Features | Balanced acc. | Macro F1 | Normal recall | False-positive rate | Top features (importance) |
|---|---|---|---|---|---|---|
| A. Behavioural features (honest) | 14 | 0.9793 | 0.9568 | 0.981 | 1.87% | srate (0.27); N_IN_Conn_P_DstIP (0.13); stddev (0.12) |
| B. Behavioural + identifiers (leaky) | 20 | 1.0000 | 0.9919 | 1.000 | 0.00% | pkSeqID (0.37); dport (0.14); daddr (0.09) |
| C. Row ID only (pkSeqID) | 1 | 1.0000 | 0.9991 | 1.000 | 0.00% | pkSeqID (1.00) |
| D. Without whole-capture connection counts | 12 | 0.8620 | 0.7638 | 0.785 | 21.50% | srate (0.32); max (0.15); stddev (0.14) |
