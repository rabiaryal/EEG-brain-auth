# EEG-Auth

Can you log in with your brain signals? And if you can, how easily can someone fake it?

I built a small EEG login system, then spent most of the project trying to break it. It works well on honest users and is easy to fool with replayed recordings or small adversarial changes to the signal. This README starts with the three things you should know before trusting any number: the **dataset**, the **limitations** and the **results**. The rest is how it works and how to run it.

(The folder on my laptop is called "ECG brain authentication" for historical reasons. The signals are EEG.)

## Dataset

- **PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)**: public, 64 channels, 160 Hz, 109 volunteers.
- I used **106 subjects**. Three (S088, S092, S100) were dropped because runs 3–7 are not recorded at 160 Hz.
- Signals are band-pass filtered at 1–40 Hz and cut into 2-second clips (64 channels × 320 samples).
- **Train runs: 1, 3, 4, 5. Test runs: 2, 6, 7.** No recording run is ever on both sides, so the model is always tested on recordings it has never seen. Totals: 22,641 training clips and 16,162 test clips.

## Limitations (please read these)

- **It's lab data from one visit.** The "sessions" in this dataset are runs recorded during the same visit, not different days. Splitting by run is better than splitting random clips, but it doesn't remove same-day similarity. A real login system faces tomorrow's brain, so real error rates would very likely be worse than mine.
- **No real hardware.** Everything is a public recording from a research-grade cap. I didn't test a consumer headset or live acquisition.
- **One random seed (42).** I have no error bars across seeds. The attack results use 200 attacker–victim pairs per setting, which helps, but small differences (a point or two) shouldn't be over-read.
- **Attacks are digital and white-box.** The attacker knows the model and edits the signal directly. I didn't test whether such a perturbation could be produced on a real person.
- **Challenge-response is a simulation.** It uses the dataset's task labels, not a real live challenge, and the task decoder is weak (46% accuracy on four classes).
- **Adversarial training doesn't fully work.** It helps only partly (see below) and did not stop a smarter attack against the embedding-based matcher used in the API demo (100% → 100% on 30 test clips, 20-subject run).
- The reference electrode (C4) is kept so the clips stay 64 channels.

## Results

All numbers are on test clips from runs never seen in training. A quick note on terms: **EER** is the error rate where false accepts equal false rejects (lower is better).

| What I measured | 20 subjects | **106 subjects** |
|---|---|---|
| Logistic regression: accuracy / EER | 95.80% / 1.77% | **91.54% / 1.98%** |
| SVM: accuracy / EER | 93.70% / 5.19% | **88.58% / 4.90%** |
| CNN: accuracy / EER | 93.99% / 1.98% | **88.34% / 3.04%** |
| CNN used by the API (embedding match): EER | 3.05% | **7.33%** |
| Random impostor accepted (at the EER threshold) | 1.98% | **3.06%** |
| **Replay attack** accepted | 98.03% | **96.96%** |
| **Adversarial attack** (PGD, ε = 0.05) accepted | 64.0% | **75.0%** |
| **Adversarial attack** (PGD, ε = 0.1) accepted | 96.0% | **94.5%** |
| Same attack after adversarial training | 92.0% | **86.5%** |
| Clean EER before → after adversarial training | 1.98% → 2.36% | **3.04% → 4.42%** |
| Replay accepted with challenge-response (1–4 recorded tasks) | 12.4 / 25.6 / 35.5 / 47.0% | **10.2 / 21.3 / 29.0 / 40.4%** |

ε is the size of the perturbation in normalised units; at ε = 0.1 it is about 14% of the signal's size.

**In plain words:**
- Identity from 2 seconds of EEG works: about 2% EER with the simplest model, even with 106 people.
- The fancy CNN did **not** beat plain band-power features with logistic regression.
- Replaying a genuine recording is accepted about 97% of the time. A static biometric can't tell live from recorded.
- A small crafted perturbation gets an impostor accepted about 94.5% of the time.
- Adversarial training lowers that to 86.5%, but costs accuracy, and at the smallest ε it made things slightly worse (23.5% → 25.5%). It's a partial fix, not a solution.

| | |
|---|---|
| ![Model comparison](results/final/fig_models.png) | ![Attack success vs epsilon](results/final/fig_epsilon_curve.png) |
| ![Adversarial training before and after](results/final/fig_before_after.png) | ![Replay with challenge-response](results/final/fig_replay_challenge.png) |

The full table is in [`results/final/final_results.md`](results/final/final_results.md) and the longer write-up is in [`report/report.md`](report/report.md).

---

## How it works

1. **Features and models.** Two classic baselines (logistic regression and an RBF-SVM on log band power in the theta, alpha, beta and gamma bands, 256 features) and a small 1D CNN (~105k parameters) on the raw clips.
2. **Verification.** For each test clip, the score for the true identity is the "genuine" score and the scores for all other identities are "impostor" scores. From these I compute ROC, AUC, EER, and false accept / false reject rates.
3. **Attacks.** A random impostor, replay of genuine clips, and targeted white-box FGSM and PGD attacks at six perturbation sizes.
4. **Defenses.** Adversarial training with PGD examples, and a simulated challenge-response against replay.
5. **Login API.** A small FastAPI service that enrolls a user and verifies later clips by cosine similarity to a per-user template, so new people can enroll without retraining.

## Run it yourself

You need Python 3.12 and roughly 10 GB of free disk for the full run (raw files are about 2 GB, plus caches). The `data/` and `models/` folders are created by the scripts and aren't in git.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -m "not slow"                        # quick tests
```

Quick run (20 subjects):

```bash
python scripts/build_dataset.py             # downloads from PhysioNet
python scripts/run_baseline.py
python scripts/run_cnn.py
python scripts/run_attacks.py
python scripts/run_defenses.py
python scripts/eval_embeddings.py --checkpoint cnn
python scripts/eval_embeddings.py --checkpoint cnn_robust
```

Full run (106 subjects):

```bash
python scripts/download_all.py              # download and check sampling rates
bash scripts/run_full.sh
python scripts/run_defenses.py --config configs/full109.yaml
python scripts/make_figures.py              # writes results/final/
```

Seeds are fixed to 42 and saved with every result. Training uses Apple MPS if available, otherwise CPU.

### Login API demo

```bash
python scripts/demo_api.py                  # enroll A, verify A, B claims A, adversarial clip
uvicorn --factory eegauth.api.app:create_app --app-dir src
```

Endpoints: `POST /enroll`, `POST /verify` (returns a JWT when accepted) and `GET /health`. Inputs are validated and each user's data is kept separate. This is a demo: the JWT secret is a config constant, and the `/token/decode` debug endpoint should never be deployed.

## What's in the repo

```
configs/        20-subject and 106-subject settings
src/eegauth/    data, features, models, training, metrics, attacks, defenses, api/
scripts/        run_*.py, make_figures.py, demo_api.py, run_full.sh
tests/          pytest suite
results/        metrics and figures (20 subjects, full109/, final/)
report/         report.md, the longer write-up
```

## Credits

Data: [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/), released under the Open Data Commons Attribution License. Please cite:

- Schalk, G. et al. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on Biomedical Engineering* 51(6), 2004.
- Goldberger, A. et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), 2000.
