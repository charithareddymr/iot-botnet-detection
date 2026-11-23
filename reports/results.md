# Results (identifier columns removed, full untouched test set)

| Model | Accuracy | Balanced acc. | Macro F1 | False-positive rate | Attack detection | Train time |
|---|---|---|---|---|---|---|
| Random Forest | 0.9962 | 0.9793 | 0.9568 | 1.87% | 99.9977% | 3s |
| AdaBoost | 0.8986 | 0.9362 | 0.8803 | 1.87% | 99.9926% | 56s |
| KNN | 0.9891 | 0.9742 | 0.9169 | 2.80% | 99.9950% | 0s |
| MLP | 0.9591 | 0.9454 | 0.7695 | 3.74% | 99.9885% | 14s |
| RNN (LSTM) | 0.8951 | 0.9318 | 0.5233 | 2.80% | 99.7718% | 46s |

## Recall per class (share of each class correctly identified)

| Model | Normal | DDoS | DoS | Reconnaissance | Theft |
|---|---|---|---|---|---|
| Random Forest | 0.981 | 0.995 | 0.998 | 0.994 | 0.929 |
| AdaBoost | 0.981 | 0.975 | 0.804 | 0.992 | 0.929 |
| KNN | 0.972 | 0.986 | 0.992 | 0.992 | 0.929 |
| MLP | 0.963 | 0.939 | 0.981 | 0.987 | 0.857 |
| RNN (LSTM) | 0.972 | 0.875 | 0.914 | 0.969 | 0.929 |
