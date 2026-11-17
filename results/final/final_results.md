| Metric | 20 subjects | all screened subjects |
|---|---|---|
| Subjects | 20 | 106 |
| Train / test clips | 4269 / 3046 | 22641 / 16162 |
| LogReg identification acc | 95.80% | 91.54% |
| LogReg EER (AUC) | 1.77% (0.9990) | 1.98% (0.9960) |
| SVM identification acc | 93.70% | 88.58% |
| SVM EER (AUC) | 5.19% (0.9885) | 4.90% (0.9869) |
| CNN identification acc | 93.99% | 88.34% |
| CNN EER (AUC), softmax score | 1.98% (0.9983) | 3.04% (0.9921) |
| CNN EER, embedding-cosine (API verifier) | 3.05% | 7.33% |
| Zero-effort impostor FAR @ EER thr. | 1.98% | 3.06% |
| Replay acceptance (unseen test recordings) | 98.03% | 96.96% |
| Replay acceptance (enrollment recordings) | 100.00% | 99.99% |
| FGSM eps=0.02 acceptance | 14.0% | 21.0% |
| PGD eps=0.02 acceptance | 14.0% | 23.5% |
| FGSM eps=0.05 acceptance | 56.0% | 56.0% |
| PGD eps=0.05 acceptance | 64.0% | 75.0% |
| FGSM eps=0.1 acceptance | 88.0% | 73.5% |
| PGD eps=0.1 acceptance | 96.0% | 94.5% |
| PGD eps=0.1 L2 perturbation / signal | 0.137 | 0.143 |
| Adv. training: clean id. acc (before -> after) | 93.99% -> 93.47% | 88.34% -> 83.36% |
| Adv. training: clean EER (before -> after) | 1.98% -> 2.36% | 3.04% -> 4.42% |
| PGD eps=0.02: before -> after adv. training | 14.0% -> 8.0% | 23.5% -> 25.5% |
| PGD eps=0.05: before -> after adv. training | 64.0% -> 56.0% | 75.0% -> 58.5% |
| PGD eps=0.1: before -> after adv. training | 96.0% -> 92.0% | 94.5% -> 86.5% |
| Task-classifier accuracy (challenge-response) | 51.7% | 46.1% |
| Replay success with challenge, k=1 | 12.4% | 10.2% |
| Replay success with challenge, k=2 | 25.6% | 21.3% |
| Replay success with challenge, k=3 | 35.5% | 29.0% |
| Replay success with challenge, k=4 | 46.9% | 40.4% |
| Attack pairs per epsilon | 50 | 200 |
