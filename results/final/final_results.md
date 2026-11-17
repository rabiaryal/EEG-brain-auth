| Metric | 20 subjects |
|---|---|
| Subjects | 20 |
| Train / test clips | 4269 / 3046 |
| LogReg identification acc | 95.80% |
| LogReg EER (AUC) | 1.77% (0.9990) |
| SVM identification acc | 93.70% |
| SVM EER (AUC) | 5.19% (0.9885) |
| CNN identification acc | 93.99% |
| CNN EER (AUC), softmax score | 1.98% (0.9983) |
| CNN EER, embedding-cosine (API verifier) | 3.05% |
| Zero-effort impostor FAR @ EER thr. | 1.98% |
| Replay acceptance (unseen test recordings) | 98.03% |
| Replay acceptance (enrollment recordings) | 100.00% |
| FGSM eps=0.02 acceptance | 14.0% |
| PGD eps=0.02 acceptance | 14.0% |
| FGSM eps=0.05 acceptance | 56.0% |
| PGD eps=0.05 acceptance | 64.0% |
| FGSM eps=0.1 acceptance | 88.0% |
| PGD eps=0.1 acceptance | 96.0% |
| PGD eps=0.1 L2 perturbation / signal | 0.137 |
| Adv. training: clean id. acc (before -> after) | 93.99% -> 93.47% |
| Adv. training: clean EER (before -> after) | 1.98% -> 2.36% |
| PGD eps=0.02: before -> after adv. training | 14.0% -> 8.0% |
| PGD eps=0.05: before -> after adv. training | 64.0% -> 56.0% |
| PGD eps=0.1: before -> after adv. training | 96.0% -> 92.0% |
| Task-classifier accuracy (challenge-response) | 51.7% |
| Replay success with challenge, k=1 | 12.4% |
| Replay success with challenge, k=2 | 25.6% |
| Replay success with challenge, k=3 | 35.5% |
| Replay success with challenge, k=4 | 46.9% |
| Attack pairs per epsilon | 50 |
