# EEG-Auth

I wanted to see whether brain signals can be used to log someone in, and how easily such a system can be fooled. So I built a small EEG verifier and then tried to break it.

It identifies people well on normal recordings (about 2% error), but it is easy to fool: replaying a genuine recording gets accepted almost every time, and a small crafted change to the signal gets an impostor accepted about 94% of the time. Adversarial training helped only a little. Details are below.

## Dataset

I used the public [PhysioNet EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/) (64 channels, 160 Hz, 109 volunteers). Three subjects (S088, S092, S100) were left out because some of their runs aren't recorded at 160 Hz, so everything below uses **106 subjects**.

Signals are filtered to 1–40 Hz and cut into 2-second clips. Training uses runs 1, 3, 4 and 5; testing uses runs 2, 6 and 7. No run appears on both sides, so the model is always tested on recordings it has never seen. That gives 22,641 training clips and 16,162 test clips.

## Results

EER is the error rate where false accepts equal false rejects (lower is better). I ran everything on a 20-subject subset first and then on all 106 subjects.

| | 20 subjects | 106 subjects |
|---|---|---|
| Logistic regression (band power): accuracy / EER | 95.80% / 1.77% | 91.54% / 1.98% |
| SVM: accuracy / EER | 93.70% / 5.19% | 88.58% / 4.90% |
| CNN: accuracy / EER | 93.99% / 1.98% | 88.34% / 3.04% |
| Random impostor accepted | 1.98% | 3.06% |
| Replay of a genuine recording accepted | 98.03% | 96.96% |
| Adversarial attack accepted (PGD, ε = 0.05) | 64.0% | 75.0% |
| Adversarial attack accepted (PGD, ε = 0.1) | 96.0% | 94.5% |
| Same attack after adversarial training (ε = 0.1) | 92.0% | 86.5% |
| EER before → after adversarial training | 1.98% → 2.36% | 3.04% → 4.42% |

What I take from this:

- The simplest model (logistic regression on band-power features) was the best. The CNN did not beat it.
- Going from 20 to 106 people costs a few points of accuracy, but the EER stays low.
- Replay can't be fixed by a better model. A recording of you looks the same as you.
- Adversarial training lowered the attack success from 94.5% to 86.5%, but it made the clean error worse, and at the smallest perturbation it was slightly worse than no defense. It is a partial fix at best.

I also simulated a challenge-response check against replay (the user is asked to do a task and the system checks it). It cut replay success to roughly 10–40%, but the task classifier is weak (46% accuracy on four tasks), so treat it as a rough idea, not a working defense.

The full table and figures are in `results/final/`, and a longer write-up is in `report/report.md`.

![Model comparison](results/final/fig_models.png)
![Attack success vs perturbation size](results/final/fig_epsilon_curve.png)

## Limitations

- **Same-visit data.** The "sessions" are runs recorded during the same visit, not different days. Splitting by run is better than splitting random clips, but real logins happen days apart, so real error rates would probably be worse than mine.
- **Lab recordings.** The data comes from a research-grade EEG cap. I didn't test a consumer headset or live recording.
- **One seed.** Everything was run with seed 42 and I have no error bars across seeds, so differences of a point or two shouldn't be over-read.
- **Digital, white-box attacks.** The attacker knows the model and edits the signal directly. I didn't test whether this could be done to a real person.
- **Weak defenses.** Adversarial training only partly helped, and in a separate test against the embedding-based matcher used in my API demo it did not help at all (100% accepted before and after, on 30 clips from the 20-subject run).
- **Challenge-response is simulated** from the dataset's task labels, not a real live challenge.

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

## Credits

Data: PhysioNet EEGMMIDB (Open Data Commons Attribution License). Please cite Schalk et al., *IEEE Transactions on Biomedical Engineering* 51(6), 2004, and Goldberger et al., *Circulation* 101(23), 2000.
