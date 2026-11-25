# EEG-Auth

I wanted to see whether brain signals can be used to log someone in, and how easily such a system can be fooled. So I built a small EEG verifier, tested it on two public datasets, and then tried to break it.

**In short.** Inside one recording visit the verifier works well (about 2% error on 106 people). It is easy to fool: replaying a genuine recording is accepted almost every time, and a small crafted change to the signal gets an impostor accepted about 94% of the time. Adversarial training helped only a little. On a second dataset recorded across several days, error jumped from 1-8% within a session to 29-39% on a different day. Enrolling on two sessions, adding some data from the new session and averaging several clips brought it down to about 19-23% at best, which is still too high for a real login.

## Dataset

I used two datasets. They answer different questions, so their numbers should not be compared directly (different headsets, different people).

| | EEGMMIDB (PhysioNet) | BED (Zenodo) |
|---|---|---|
| Question it answers | How good is the verifier, and how easy is it to attack? | Does it still work on a different day? |
| People | 109 volunteers, **106 used** | 21 volunteers |
| Headset | 64 channels, research cap, 160 Hz | 14 channels, Emotiv EPOC+ consumer headset, 256 Hz (resampled to 160 Hz) |
| Sessions | One visit (runs recorded during the same visit) | 3 sessions, one week apart |
| Signal used | Motor movement / imagery runs | Resting state (eyes open / eyes closed) only |
| Access | Public | Restricted: not in this repo, never committed |
| Link | [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) | [BED, Zenodo 4309471](https://zenodo.org/records/4309471) |

**More detail.** EEGMMIDB: 106 subjects, runs 1/3/4/5 for training and 2/6/7 for testing (22,641 / 16,162 clips of 2 s, 1-40 Hz). BED: resting state only, enrolled on one or two sessions and tested on another; hyper-parameters tuned inside session-1 training data only. Full protocols are in [`report/report.md`](report/report.md).

EER (equal error rate) is the error rate where false accepts equal false rejects. Lower is better, and 50% is chance.

## Results

Everything below is summarised; **all tables, figures and protocol details are in [`report/report.md`](report/report.md)**.

### EEGMMIDB: verifier, attacks and defenses (7 experiments)

| Experiment | 20 subjects | 106 subjects |
|---|---|---|
| A1 Logistic regression: accuracy / EER | 95.80% / 1.77% | 91.54% / 1.98% |
| A1 SVM | 93.70% / 5.19% | 88.58% / 4.90% |
| A1 CNN | 93.99% / 1.98% | 88.34% / 3.04% |
| A1 CNN, embedding-cosine matcher (API demo): EER | 3.05% | 7.33% |
| A2 Random impostor accepted | 1.98% | 3.06% |
| A3 Replay of a genuine recording accepted | 98.03% | 96.96% |
| A4 Adversarial attack (PGD) accepted, ε = 0.05 / 0.1 | 64% / 96% | 75.0% / 94.5% |
| A5 After adversarial training, ε = 0.1 | 96% → 92% | 94.5% → 86.5% |
| A5 Clean EER before → after adversarial training | 1.98% → 2.36% | 3.04% → 4.42% |
| A6 Replay success with challenge-response (simulated) | 12-47% | 10-40% |
| A7 Scaling 20 → 106 subjects | | accuracy drops 4-6 points, EER stays near 2% |

![Model comparison](results/final/fig_models.png)
![Attack success vs perturbation size](results/final/fig_epsilon_curve.png)

### BED: does it work on a different day? (10 experiments)

EER %, 21 subjects:

| Test | Logistic regression | Riemannian | Template matching | CNN (5 seeds) |
|---|---|---|---|---|
| Same session | 2.3 | 1.2 | 5.8 | 7.7 |
| 1 week later (enrol on 1 session) | 34.5 | 35.6 | 34.0 | 29.3 |
| 2 weeks later (enrol on 1 session) | 38.9 | 38.4 | 36.2 | 33.6 |

![BED EER by time gap](results/bed/fig1_time_gap.png)

One line per experiment (tables for each are in the report):

| Experiment | What I tested | Result |
|---|---|---|
| B1/B2 | Same session vs a different day | 1-8% vs 29-39% |
| B3 | Enrol on two sessions instead of one | Lower EER in all 12 cases (0.5 to 12.1 points); best CNN on session 2, 31.3% → 19.2% |
| B4 | Add clips from the new session | 5 clips (10 s): LogReg 40.5% → 35.2%; 80 clips (~2.7 min): template 35.8% → 28.1% |
| B5 | Session alignment (z-score, Euclidean, Riemannian) | No consistent gain; Euclidean alignment hurt Riemannian (32.1% → 38.7%) |
| B6 | Eyes-closed vs eyes-open vs both | No consistent difference |
| B7 | Average 1, 3, 5, 10 clips per login | About 3 points better with two enrolment sessions (template 26.6% → 23.1%), little with one |
| B8 | Which frequency bands | All four bands best; alpha only worse (35.7% → 41.8%) |
| B9 | Impostors never enrolled | 26.9% EER, about the same as normal (26.6%) |
| B10 | Step-by-step ladder on session 3 | LogReg 40.5% → 31.0%, Riemannian 39.6% → 37.1%, CNN 33.9% → 32.1%, template 35.8% → 36.7% (worse) |

![Improvement ladder](results/bed/fig5_ladder.png)

## Findings

**On EEGMMIDB (one visit)**
1. **A simple model wins.** Logistic regression on band power was the best verifier (EER 1.98% on 106 people). The CNN did not beat it.
2. **More people costs a little.** Going from 20 to 106 subjects lowers accuracy by 4-6 points, but the EER stays low.
3. **Replay can't be fixed by a better model.** A recording of you looks the same as you (97-98% accepted).
4. **Small crafted changes work very well for an attacker.** A perturbation of about 14% of the signal size (ε = 0.1) got an impostor accepted 94.5% of the time.
5. **Adversarial training is a partial fix.** It lowered attack success (94.5% to 86.5%) but raised clean error (3.04% to 4.42%), was slightly worse at the smallest ε, and gave no protection to the embedding matcher.
6. **Challenge-response looks promising but is unproven.** In simulation it cut replay success to roughly 10-47%, limited by a weak task classifier.

**On BED (different days)**
7. **Across days it breaks.** Within a session the error is 1-8%; on a different day it is 29-39% with one enrolment session (chance is 50%). The 2% on EEGMMIDB is optimistic for real use.
8. **It is not only elapsed time.** Pairs involving session 1 are worse than the pair session 2 and 3 (CNN: 35.0% for s1 to s3, 23.4% for s2 to s3), so session 1 simply differs from the others.
9. **Enrolling on two sessions helps the most.** It lowered EER in all 12 model/session cases (by 0.5 to 12.1 points). Best: CNN on session 2, 31.3% to 19.2%.
10. **A little data from the new session helps.** 5 clips (10 seconds) per user took logistic regression from 40.5% to 35.2%; 80 clips (about 2.7 minutes) took template matching from 35.8% to 28.1%.
11. **Averaging clips helps modestly.** Fusing 5-10 clips lowered EER by about 3 points when enrolled on two sessions (template matching 26.6% to 23.1%), and little with one enrolment session.
12. **Several plausible fixes did not help.** Per-session alignment (Euclidean alignment made the Riemannian models worse: 32.1% to 38.7%), eyes-closed-only or eyes-open-only data (no consistent difference) and alpha-only features (35.7% to 41.8%).
13. **Unseen impostors are not harder.** Template matching with never-enrolled impostors scored 26.9% EER, about the same as the normal test (26.6%).
14. **Nothing reached a usable error rate.** The best results were about 19-23% EER (CNN 19.2% on session 2 and 22.1% on session 3; template matching with two sessions and fusion 23.1%).

## Limitations

- **Same-visit data (EEGMMIDB).** The "sessions" are runs from one visit, not different days. BED shows real cross-day error is much higher.
- **Two datasets, not comparable.** Different headsets, people and tasks.
- **BED is small.** 21 subjects, resting state only, 3 sessions. Only the CNN has error bars (5 seeds, four of the ten experiments); the other models ran once. Attacks and defenses were not run on BED.
- **Fixes were tested on the same three sessions**, so they show what helps here, not what will hold on new data.
- **One seed on EEGMMIDB, digital white-box attacks, simulated challenge-response.** Differences of a point or two shouldn't be over-read.

See the report for the full list.

## Running it

```bash
pip install -r requirements.txt
python scripts/build_dataset.py        # downloads the data (20 subjects)
python scripts/run_baseline.py
python scripts/run_cnn.py
python scripts/run_attacks.py
python scripts/run_defenses.py
```

For all 106 subjects, run `bash scripts/run_full.sh` and then `python scripts/run_defenses.py --config configs/full109.yaml` and `python scripts/make_figures.py`. This needs about 10 GB of free disk. There is also a small FastAPI login demo: `python scripts/demo_api.py`.

BED needs the restricted data: upload the `RAW_PARSED` `.mat` files to Google Drive, open `notebooks/colab_bed.ipynb` in Colab, set `DRIVE_BED_DIR`, and run all. A longer write-up is in `report/report.md`, and `GUIDE.md` explains the project and the notebook step by step.

BED needs the restricted data: upload the `RAW_PARSED` `.mat` files to Google Drive, open `notebooks/colab_bed.ipynb` in Colab, set `DRIVE_BED_DIR`, and run all. `GUIDE.md` explains the project and the notebook step by step.

## Credits

Data: PhysioNet EEGMMIDB (Open Data Commons Attribution License). Please cite Schalk et al., *IEEE Transactions on Biomedical Engineering* 51(6), 2004, and Goldberger et al., *Circulation* 101(23), 2000. For BED, cite Arnau-González et al., *IEEE Internet of Things Journal* 8(15), 2021.
