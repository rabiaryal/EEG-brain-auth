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

## 6. Cross-day evaluation on BED (21 subjects)

EEGMMIDB cannot show what happens across days, so I also ran the BED dataset (Arnau-González et al., IEEE IoT Journal 2021; Emotiv EPOC+, 14 channels, 256 Hz resampled to 160 Hz, 3 sessions one week apart). BED is restricted, so only metrics and figures are kept (`results/bed/`). Resting-state events only, 1-40 Hz, 2 s clips. Ten experiments were run from `notebooks/colab_bed.ipynb`; the complete tables are in `results/bed/results.md` and `results.csv`.

**Protocol.** Hyper-parameters were tuned on a time-block validation split of session-1 training data only. Scalers and any alignment used only training data or unlabelled clips of the relevant session. The CNN used 5 seeds (mean ± std); logistic regression, the Riemannian model and the template matcher are deterministic and ran once. Models: logistic regression on band power, Riemannian tangent space + logistic regression, Riemannian template matching (cosine similarity to a per-user mean), and the EEGNet-style CNN (S1, S2, S3, S10 only).

**History.** A first run (train on the first 70% of session 1; test on the rest of session 1, session 2 and session 3) gave same-session EER of 1.2-8.3% and 34-40% on sessions 2 and 3 for logistic regression, Riemannian and the CNN. That was the starting point. Two follow-up ideas (per-session alignment, enrolling on sessions 1+2) were then tried in a separate run, and the notebook was rebuilt to test them and several more in one place. The numbers below come from the rebuilt notebook, where each enrolment uses a whole session, so they differ slightly from the first run.

### 6.1 Headline: error grows with the time gap (mean EER %)

| Test | LogReg | Riemannian | Template | CNN |
|---|---|---|---|---|
| Same session | 2.3 | 1.2 | 5.8 | 7.7 |
| 1-week gap | 34.5 | 35.6 | 34.0 | 29.3 |
| 2-week gap | 38.9 | 38.4 | 36.2 | 33.6 |

![EER by time gap](../results/bed/fig1_time_gap.png)

The gap groups mix several enrol/test pairs, and the pairs are not equal: tests involving session 1 are worse than the pair session 2 and 3 (for example the CNN is 23.4% for s2 to s3 but 35.0% for s1 to s3). So the cause is not only elapsed time; session 1 differs from the others. Per-pair numbers are in `results.md` (S2).

### 6.2 What helped

- **Enrol on two sessions (S3).** EER on the held-out session fell in all 12 model/session cases, by 0.5 to 12.1 points. Largest: CNN on session 2, 31.3% to 19.2%. The CNN reached 22.1% on session 3 and 27.8% on session 1.
- **Some data from the new session (S4).** With session 1 as the base, adding clips per user from session 2 and testing on session 3: 5 clips (10 s) took logistic regression from 40.5% to 35.2% and Riemannian from 39.6% to 34.9%. At 80 clips (about 2.7 min) template matching went from 35.8% to 28.1%, logistic regression to 32.4%, Riemannian to 31.0%. Template matching gained little from the first few clips and more from larger amounts.
- **Score fusion (S7).** Averaging scores over k consecutive clips: enrolled on two sessions, logistic regression 32.4% to 29.0% (k=5), template matching 26.6% to 23.1% (k=10). With one enrolment session the gain on session 3 was 0-2 points (logistic regression none).

![Enrolment size](../results/bed/fig3_enrol_size.png)

### 6.3 What did not help

- **Session alignment (S5).** Feature z-scoring per session was neutral to slightly helpful for logistic regression (single enrolment, session 2: 35.7% to 34.2%; session 3: 40.5% to 39.4%) and slightly harmful when enrolled on two sessions (32.4% to 33.6%). Euclidean alignment hurt the Riemannian models (multi-session: 32.1% to 38.7%, template 26.6% to 37.5%). Riemannian recentering was roughly neutral. Feature z-scoring does not touch the Riemannian/template inputs, so those columns match "none" by construction.
- **Resting condition (S6).** Eyes-closed vs eyes-open vs both: differences of a few points in either direction, no consistent winner. The reported "alpha is most stable" did not show up here as an advantage.
- **Band-power set (S8).** All four bands (35.7% single, session 2) beat alpha only (41.8%), theta+alpha (41.3%) and alpha+beta (38.0%).

### 6.4 Open-set check (S9)

Impostors were people never enrolled (5 random splits: two thirds enrolled, one third unknown). Template matching: 41.1 ± 2.4% (s1 to s2), 35.1 ± 3.1% (s1 to s3), 26.9 ± 1.8% (s1+s2 to s3). The last is close to the ordinary zero-effort result (26.6%), so unseen impostors are not harder here.

### 6.5 Improvement ladder, tested on session 3 (steps chosen in advance)

| Step | LogReg | Riemannian | Template | CNN |
|---|---|---|---|---|
| 0 baseline: enrol session 1 | 40.5 | 39.6 | 35.8 | 33.9 ± 0.9 |
| 1 + per-session alignment (EA) | 39.4 | 42.8 | 41.3 | 34.7 ± 1.0 |
| 2 + enrol on sessions 1+2 | 33.6 | 38.7 | 37.5 | 33.4 ± 0.7 |
| 3 + fuse 5 clips per attempt | 31.0 | 37.1 | 36.7 | 32.1 ± 0.9 |

![Improvement ladder](../results/bed/fig5_ladder.png)

Alignment made three of four models worse; enrolment on two sessions and fusion are the useful steps, mainly for logistic regression. The ladder keeps EA in steps 2-3, which is why the Riemannian and template numbers there are worse than the multi-session numbers in 6.2 (26.6% for template without EA). Best single results in this study: CNN on session 2 after two-session enrolment (19.2%), CNN on session 3 (22.1%), and template matching with two sessions and k=10 fusion (23.1%).

**Reading it.** Within one visit everything works (EER 1-8%). Across days EER is 29-39% with one enrolment session and about 19-37% with two. Neither alignment, eye condition nor band choice closes the gap. Adding enrolment data from more sessions or from the new session helps most, but no setting here reaches an error rate suitable for a login system.

## 7. Limitations
- Lab data, no consumer hardware, no live acquisition. The EEGMMIDB runs come from the same visit, so train/test separation reduces but does not remove within-day correlation; the BED run in section 6 confirms that cross-day error is far higher.
- BED: only 21 subjects, resting state only, and only three sessions, so the fixes were compared on the same sessions they were evaluated on. Deterministic models have no error bars; the CNN (5 seeds) ran in four of the ten experiments. Adversarial attacks and defenses were not run on BED.
- 106 of 109 subjects (3 excluded for sampling rate). Single seed (42); no confidence intervals except through the larger attack-pair count.
- Adversarial attacks are digital, white-box, and applied directly to the clip; physical-world feasibility is not evaluated.
- The challenge-response is a simulation based on dataset task labels.
- The reference electrode (C4) is kept in the 64-channel input.

## 8. Future work
Test on more sessions or subjects, adaptive template updating, and a pairwise (siamese) verifier; attacks and defenses on BED; embedding-level adversarial training or detection; certified/randomised-smoothing defenses; a real challenge-response protocol with a stronger task decoder; confidence intervals across seeds.
