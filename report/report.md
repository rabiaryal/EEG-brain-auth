# EEG-Based User Authentication: Spoofing and Adversarial-Robustness Evaluation

## 1. Problem
Can short scalp-EEG clips verify identity, and does such a verifier survive realistic attacks? We build a verifier, attack it (zero-effort impostor, replay, targeted adversarial perturbations), add defenses (adversarial training, a simulated challenge-response), and report all results, including negative ones. Every number below is read from JSON files written by the scripts in `scripts/` (see `results/final/final_results.md`).

## 2. Dataset
PhysioNet EEGMMIDB: 64 channels, 160 Hz, 109 volunteers, runs 1–7 used. Recordings were screened; S088, S092 and S100 were excluded because runs 3–7 are not 160 Hz, leaving **106 subjects**. Signals are band-passed 1–40 Hz (FIR) and cut into non-overlapping 2 s clips (64×320). Full set: 22,641 train and 16,162 test clips. A 20-subject subset is also reported.

## 3. Methods
- **Protocol (anti-leakage):** train/enrollment runs {1,3,4,5}; test runs {2,6,7}. No run is on both sides (enforced in code and tested). Scalers and per-channel normalisers are fitted on train only; early stopping uses a validation split of train runs.
- **Baselines:** Welch log band power (theta/alpha/beta/gamma × 64 channels = 256 features) + StandardScaler + LogReg or RBF-SVM.
- **CNN:** compact EEGNet-style 1D CNN on raw clips (~105k parameters), Adam, early stopping.
- **Verification:** genuine score = score of the true claimed identity; impostor scores = all other claimed identities. EER from the interpolated ROC. The API verifier uses cosine similarity to a per-user template embedding so new users can enroll without retraining.
- **Attacks:** zero-effort impostor; replay of genuine clips; targeted white-box FGSM/PGD on the CNN, with ε in per-channel-normalised units (200 attacker/victim pairs per ε in the 106-subject run).
- **Defenses:** PGD adversarial training (clean + adversarial examples, ε=0.1); challenge-response simulation using dataset task labels (clearly a simulation, not a live challenge).

## 4. Results (test runs, never seen in training)

| Metric | 20 subj. | 106 subj. |
|---|---|---|
| LogReg identification acc / EER | 95.80% / 1.77% | 91.54% / 1.98% |
| SVM identification acc / EER | 93.70% / 5.19% | 88.58% / 4.90% |
| CNN identification acc / EER (softmax) | 93.99% / 1.98% | 88.34% / 3.04% |
| CNN EER, embedding-cosine (API verifier) | 3.05% | 7.33% |
| Zero-effort FAR at EER threshold | 1.98% | 3.06% |
| Replay acceptance (unseen test recordings) | 98.03% | 96.96% |
| PGD ε=0.05 / 0.1 acceptance | 64% / 96% | 75.0% / 94.5% |
| Adv. training: PGD ε=0.1 acceptance | 96% → 92% | 94.5% → 86.5% |
| Adv. training: clean accuracy / EER | 93.99→93.47% / 1.98→2.36% | 88.34→83.36% / 3.04→4.42% |
| Replay success with challenge, k=1…4 | 12.4/25.6/35.5/47.0% | 10.2/21.3/29.0/40.4% |

Figures: `results/final/fig_models.png`, `fig_epsilon_curve.png`, `fig_before_after.png`, `fig_replay_challenge.png`; per-run ROC and loss curves in `results/full109/figures/`.

## 5. Discussion
**Scaling.** Going from 20 to 106 subjects costs 4–6 points of identification accuracy for every model, while the EER of the band-power LogReg stays near 2%. The simple baseline is again the best verifier; the CNN is not better on this protocol. The CNN's embedding-cosine EER degrades more (3.05% → 7.33%), the price of open-set template matching.

**Zero-effort impostors** are accepted at about the EER (≈3%).

**Replay** succeeds ~97–100% against any static biometric: a recorded genuine clip is indistinguishable from a live one, and no model change fixes this. Only a liveness mechanism can help.

**Adversarial perturbations** are very effective: PGD at ε=0.1 (L2 perturbation ≈14% of signal norm) gets ~94.5% acceptance against the undefended CNN.

**Adversarial training** helps only partly at 106 subjects: acceptance drops 75.0% → 58.5% at ε=0.05 and 94.5% → 86.5% at ε=0.1, but at ε=0.02 it is slightly *worse* (23.5% → 25.5%), and clean accuracy falls 5 points and EER rises 1.4 points. On 20 subjects (50 pairs) the gain was within noise. Against an adaptive attack on the embedding matcher actually used by the API, adversarial training gave no protection (100% → 100%, 20-subject demo, n=30).

**Challenge-response** (simulated) reduces replay success to 10–40% depending on how many challenge responses an attacker has recorded, but the task classifier is weak (46% accuracy over 4 classes), so this is a proof of concept and not a deployable defense.

## 6. Limitations
- Lab data, no consumer hardware, no live acquisition. Runs come from the same visit, so train/test separation reduces but does not remove within-day correlation; real cross-day EERs would be worse.
- 106 of 109 subjects (3 excluded for sampling rate). Single seed (42); no confidence intervals except through the larger attack-pair count.
- Adversarial attacks are digital, white-box, and applied directly to the clip; physical-world feasibility is not evaluated.
- The challenge-response is a simulation based on dataset task labels.
- The reference electrode (C4) is kept in the 64-channel input.

## 7. Future work
Cross-day data; embedding-level adversarial training or detection; certified/randomised-smoothing defenses; a real challenge-response protocol with a stronger task decoder; confidence intervals across seeds.
