# EEG-Auth: EEG-Based User Authentication with Spoofing and Adversarial-Robustness Evaluation

**Status: work in progress (Phases 0–3 complete). Final results table below reflects only numbers produced by code actually run in this repo.**

## Summary

EEG signals are as unique as fingerprints, which makes them a candidate biometric — but a biometric system is only trustworthy if it survives attacks. This project builds an identity verifier on short (2 s) clips of raw scalp EEG from the public [PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, 64 channels, 160 Hz), then attacks it with zero-effort impostors, replay, and targeted adversarial perturbations (FGSM/PGD), adds defenses (adversarial training, challenge-response), and reports every result honestly. **All train/test evaluations use disjoint recording runs** — test clips come from runs the model never saw — and all normalisation statistics are fitted on training data only.

## Current results (20 subjects, test runs {2, 6, 7}, 3,046 test clips)

| Model | Identification accuracy | Verification EER | AUC |
|---|---|---|---|
| Baseline: LogReg (256 Welch band-power features) | **95.80%** | **1.77%** | 0.9990 |
| Baseline: RBF-SVM (C=10) | 93.70% | 5.19% | 0.9885 |
| CNN (compact EEGNet-style, raw 64×320 clips, ~99k params) | 93.99% | 1.98% | 0.9983 |

Chance level for 20-subject identification is 5%. Verification protocol: for each test clip, the genuine score is the classifier score of the true claimed identity; impostor scores are the scores of every other claimed identity (57,874 zero-effort impostor attempts). EER is interpolated on the ROC curve; chance on the anti-diagonal.

Honest observation: the simple LogReg baseline is still the best overall model. Validation accuracy of the CNN reaches 100% (validation clips come from the same *runs* as training clips), while cross-run test accuracy is ~94% — a reminder that same-session generalisation overestimates deployment performance. Attack robustness (Phase 4) is expected to be where model choice matters more.

## Repository layout

```
configs/default.yaml      subjects, runs, epochs, filter band, seeds, paths
src/eegauth/
  data.py                 download EEGMMIDB, filter, epoch, session-based split
  features.py             Welch log band-power features
  models.py               compact 1D CNN
  train.py                baselines, per-channel normalizer, CNN training loop
  metrics.py              ROC / EER / FAR / FRR verification protocol
  attacks.py defenses.py  (Phases 4–5)
  api/                    (Phase 6: FastAPI login demo)
scripts/                  build_dataset.py, run_baseline.py, run_cnn.py, ...
tests/                    pytest (fast unit tests + slow real-data tests)
results/                  JSON metrics + figures
report/                   (Phase 8: research write-up)
```

## Reproduce

```bash
uv venv --python 3.12 .venv            # or python -m venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -m "not slow"        # fast tests
.venv/bin/python scripts/build_dataset.py             # downloads EEGMMIDB (~350 MB for 20 subj)
.venv/bin/python -m pytest tests -m slow              # real-data tests
.venv/bin/python scripts/run_baseline.py
.venv/bin/python scripts/run_cnn.py
```

Seeds (python/numpy/torch = 42) are fixed in the config and logged with every result.

## Limitations (honest)

- Public lab dataset (EEGMMIDB) recorded in controlled conditions; no consumer EEG hardware, no live acquisition.
- "Sessions" here are *runs* within the same recording visits; train/test run separation reduces but does not eliminate within-day correlation. Expect optimistic EERs relative to true cross-day deployment.
- 20 subjects so far; scaling to all 109 is planned (Phase 7).
- EDF channel C4 in EEGMMIDB was the physical reference electrode; it is kept for the specified 64-channel format (minimum observed training std was 2.07e-05 V, so normalisation remains well-defined).
