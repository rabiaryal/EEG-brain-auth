# Project guide: EEG authentication, and how the Colab notebook works

This guide is for understanding. The short, public-facing summary is in `README.md`. Here I explain the ideas, the code layout, the results, and then the Colab notebook step by step.

---

## 1. The idea in plain words

A password proves you *know* something. EEG authentication tries to prove who you *are* from your brain signals. The system works like this:

1. **Enrol:** record a person's EEG, cut it into 2-second clips, and train a model to recognise them.
2. **Log in:** record a new clip. The model says "this is user X" or "not X".
3. **Attack:** try to fool the model, because a login system is only useful if it is hard to cheat.

The project does all three parts: build the verifier, measure how good it is, then attack it and try to defend it.

## 2. Key terms

| Term | Meaning |
|---|---|
| **Clip** | A 2-second slice of EEG. The unit the model sees. |
| **Identification accuracy** | Given a clip, how often the model names the right person out of all users. |
| **Verification** | "Is this clip really user X?" A yes/no decision per claim. |
| **FAR** | False Accept Rate: an impostor is let in. |
| **FRR** | False Reject Rate: the real user is turned away. |
| **EER** | Equal Error Rate: the point where FAR = FRR. One number for system quality, lower is better. |
| **Zero-effort impostor** | Someone who just tries to log in as another person with their own brain signal. |
| **Replay attack** | Playing back a stored genuine recording. |
| **FGSM / PGD** | Adversarial attacks: tiny, crafted changes to the signal that flip the model's decision. ε is the size of the change. |
| **Adversarial training** | Training the model on attacked examples so it resists them. |
| **Challenge-response** | The system asks for a random task (e.g. open/close eyes), so a replayed recording of a different task fails. |

## 3. Datasets

### EEGMMIDB (used for all reported results)
- Public, from PhysioNet. 109 volunteers, 64 channels, 160 Hz.
- 106 subjects used (S088, S092, S100 excluded: runs not at 160 Hz).
- Filter 1–40 Hz, 2 s clips.
- **Split by run, never by clip:** train on runs 1, 3, 4, 5 (22,641 clips), test on runs 2, 6, 7 (16,162 clips). 15% of training clips are held out for early stopping, so the test runs are never used for tuning.
- Weakness: everything is from one visit, so it does not tell us about performance across days.

### BED (run on Colab; results in `results/bed/`)
- Arnau-González et al., IEEE Internet of Things Journal, 2021. Zenodo record 4309471.
- 21 subjects, Emotiv EPOC+ (consumer headset), 256 Hz, 14 channels, **3 sessions one week apart**.
- That is exactly what EEGMMIDB lacks: it lets us test **across days**.
- **Restricted dataset.** Never commit raw or processed BED data to git. Keep it on your private Drive.

## 4. Repository layout

```
src/eegauth/        library code (data loading, features, models, training, attacks, defenses)
scripts/            command-line entry points (build dataset, run baseline/CNN/attacks/defenses, figures)
configs/            YAML settings (20-subject and full-106 runs)
results/final/      final tables and figures
report/             longer report + the 2-page summary PDF
notebooks/          colab_bed.ipynb (BED on Google Colab)
README.md           short public summary
GUIDE.md            this file
```

## 5. Models and experiments

- **Logistic regression / SVM** on Welch band-power features (log power per channel per band; 256 features). Simple and fast.
- **Small 1D CNN** (`EEGCNNSmall`, about 105k parameters) on the raw clip.
- **Embedding verifier** (used in the FastAPI demo): cosine similarity between CNN embeddings, with a JWT issued on success.

Experiments, in order: baseline models → CNN → attacks (impostor, replay, FGSM/PGD) → defenses (adversarial training, simulated challenge-response).

## 6. Results so far (106 subjects, EEGMMIDB)

| Measure | Result |
|---|---|
| Logistic regression: accuracy / EER | 91.54% / 1.98% |
| SVM | 88.58% / 4.90% |
| CNN | 88.34% / 3.04% |
| Random impostor accepted | 3.06% |
| Replay accepted | 96.96% |
| PGD attack accepted, ε = 0.05 / 0.1 | 75.0% / 94.5% |
| After adversarial training, ε = 0.1 | 86.5% (clean EER 3.04% → 4.42%) |

Takeaways: the simplest model is the best; replay cannot be fixed by a better model; adversarial training only partly helps. 

**BED (21 subjects, across days), EER:** same session 1.2-7.7%, different day 29-39% with one enrolment session, about 19-37% with two sessions. Details in `results/bed/results.md` and `report/report.md` section 6.

## 7. The Colab notebook (`notebooks/colab_bed.ipynb`)

### Why a notebook
BED is restricted, so its data must not live in the git repo. Colab gives a free GPU, and Google Drive keeps the data private. All BED-specific code is **inside the notebook** (cell 5 writes it out as a Python module at run time), so the repository stays free of BED code and data.

### The flow

```
Drive (BED .mat files) ──► find files ──► inspect ──► preprocess to clips ──► train/test protocols ──► results table + figure
                                                         (saved to Drive)                                (saved to Drive)
```

### Cell by cell

| Cell | What it does |
|---|---|
| 0 | Instructions (markdown). |
| 1 | Checks that a GPU is available and how much disk there is. |
| 2 | Clones this repo into `/content/ECG-brain-auth` and installs `mne`, `pyriemann`, `mat73`, `pyyaml`. |
| 3 | Mounts Google Drive (one authorisation click). |
| 4 | **Config**: all the settings you may edit (see below). |
| 5 | `%%writefile` writes `src/eegauth/bed.py` inside Colab. This is the BED module: loading, event parsing, clip extraction, splits, models, protocols, reporting. |
| 6 | Imports (or reloads) that module. |
| 7 | Finds the BED files in Drive. Uses `DRIVE_BED_DIR` if it contains `.mat` or `.zip`; otherwise searches Drive up to depth 6. Zips are copied to Colab's local disk and only `.mat` files are extracted, so the originals on Drive are never touched. Builds a map `{(subject, session): path}`. |
| 8 | Opens one `.mat` and prints its shape, sampling rate, channel columns, events and label counts, and checks the resting-state rules match. **Read this output.** |
| 9 | Preprocesses every subject and saves one `.npz` each to `MyDrive/eeg_auth_bed/`. Resumable: if Colab disconnects, run again and finished subjects are skipped. |
| 10 | Prints a summary of the prepared dataset. |
| 11 | Runs the protocols and saves results to `MyDrive/eeg_auth_bed/results/`. |
| 12 | Shows the table and the figure. |
| 13 | Optional: zips the metrics and figure for download. |

### Config values (cell 4)

| Setting | Default | Meaning |
|---|---|---|
| `DRIVE_BED_DIR` | `/content/drive/MyDrive/BED` | Folder where you uploaded the BED files. **Change this to yours.** |
| `WORK` | `/content/drive/MyDrive/eeg_auth_bed` | Where processed data and results go. |
| `N_SUBJECTS` | 21 | How many subjects to use. |
| `N_SEEDS` | 5 | Seeds for the CNN (mean ± std). |
| `TARGET_FS` / `BAND` / `CLIP_S` | 160 / (1, 40) / 2 | Resample rate, band-pass in Hz, clip length in seconds. |
| `TRAIN_FRAC` / `VAL_FRAC` | 0.7 / 0.25 | Time-based split fractions. |
| `MAX_EPOCHS` | 60 | CNN training cap (early stopping applies). |
| `REST_RULES` | `eyes_closed: clos`, `eyes_open: open` | Regexes that pick resting-state events by label text. **May need editing** after cell 8. |

### Preprocessing (inside the module)
1. Load `sN_sM.mat` (`recording`: time × 17 columns; `events`: start/end times plus info).
2. Keep the 14 EEG channels; band-pass 1–40 Hz on the whole recording.
3. Cut out the resting-state events (eyes-open / eyes-closed first, as they are the least task-dependent), trim 1 s from each end.
4. Resample 256 Hz → 160 Hz, cut 2 s clips.

The stimulus labels in BED are not documented, so the notebook discovers them at run time (cell 8) instead of guessing.

### Evaluation protocol
- **Train:** the first 70% (by time) of each session-1 segment.
- **Same-session test:** the later part of session 1, after a one-clip gap so neighbouring clips do not leak.
- **1-week test:** session 2. **2-week test:** session 3.
- **Tuning:** only on a time-block validation split inside the training data.
- **Scalers** are fitted on training data only.
- **Models:** logistic regression on band power; Riemannian (pyriemann tangent space + logistic regression); an EEGNet-style CNN. The CNN is run with 5 seeds (mean ± std); the deterministic models run once.
- **Output:** a table `same-session EER | 1-week EER | 2-week EER` per model, plus a figure.

Result: error jumps from 1-8% (same session) to 29-39% on a different day. Enrolling on two sessions, a little data from the new session, and averaging several clips help; alignment, eye condition and alpha-only features did not.

## 8. How to run the notebook

1. Upload BED files to Drive (e.g. `MyDrive/BED/`) and wait for the sync to finish.
2. In Colab: *File → Upload notebook* → `notebooks/colab_bed.ipynb`.
3. *Runtime → Change runtime type → GPU*.
4. Edit `DRIVE_BED_DIR` in cell 4.
5. Run cells 0–8 first and read the cell-8 output. If resting-state events are not found, fix `REST_RULES`.
6. *Runtime → Run all*. Results appear in `MyDrive/eeg_auth_bed/results/` (`bed_results.json`, `bed_table.md`, `bed_eer.png`).

## 9. Honest status and cautions

- The notebook was first tested on synthetic data, then run on the real BED files; the real numbers are in `results/bed/`. Only 21 subjects, resting state only.
- Label names in the real files may differ from my regex guesses; cell 8 is the check.
- Processed clips on Drive are derived from restricted data. Keep that folder private.
- Never put a Zenodo token in a notebook; use Colab Secrets if you ever download directly.
- Citation: Arnau-González et al., IEEE IoT Journal, 2021 (BED); Schalk et al., IEEE TBME 51(6), 2004 and Goldberger et al., Circulation 101(23), 2000 (EEGMMIDB).
