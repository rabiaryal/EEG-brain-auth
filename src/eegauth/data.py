"""Data pipeline for the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB).

Pipeline: download EDF files via ``mne.datasets.eegbci`` -> standardize channel
names/montage -> band-pass filter -> cut into fixed-length 2 s non-overlapping
clips (shape ``n_clips x n_channels x n_samples`` = ``n x 64 x 320`` at 160 Hz)
-> split clips into train/test by *run* so that no recording contributes clips
to both sides (no data leakage).

Caching is two-level:
1. per (subject, run) filtered clip arrays as ``.npz`` in the cache dir;
2. the full train/test split as a single ``.npz`` keyed by a hash of the split
   parameters, so later phases can reload the exact same dataset instantly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import mne
import numpy as np
from mne.datasets import eegbci

__all__ = [
    "EEGClips",
    "clip_arrays_from_raw",
    "load_run_clips",
    "load_run_task_clips",
    "split_by_runs",
    "build_dataset",
    "dataset_stats",
]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class EEGClips:
    """Fixed-length EEG clips with provenance.

    Attributes:
        X: float32 array of shape (n_clips, n_channels, n_samples).
        y: int array of shape (n_clips,) with subject-class indices.
        subject_ids: int array of shape (n_clips,) with PhysioNet subject numbers.
        run_ids: int array of shape (n_clips,) with EEGMMIDB run numbers (1-14).
    """

    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    run_ids: np.ndarray

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        assert self.y.shape == (n,), "y must have one label per clip"
        assert self.subject_ids.shape == (n,), "subject_ids must align with X"
        assert self.run_ids.shape == (n,), "run_ids must align with X"

    @property
    def n_clips(self) -> int:
        return int(self.X.shape[0])


# --------------------------------------------------------------------------- #
# Per-run loading (download -> filter -> epoch), with caching
# --------------------------------------------------------------------------- #
def clip_arrays_from_raw(
    raw: mne.io.BaseRaw, length_s: float, overlap_s: float = 0.0
) -> np.ndarray:
    """Cut a raw recording into fixed-length clips.

    Args:
        raw: preloaded MNE raw object (already filtered, EEG channels only).
        length_s: clip duration in seconds.
        overlap_s: overlap between consecutive clips in seconds (0 = disjoint).

    Returns:
        float32 array of shape (n_clips, n_channels, n_samples). Any trailing
        samples that do not fill a complete clip are dropped.
    """
    sfreq = raw.info["sfreq"]
    n_samples = int(round(length_s * sfreq))
    step = int(round((length_s - overlap_s) * sfreq))
    if n_samples <= 0 or step <= 0:
        raise ValueError(f"Invalid epoch geometry: length={length_s}s, overlap={overlap_s}s")
    data = raw.get_data()  # (n_channels, n_times), float64
    n_times = data.shape[1]
    if n_times < n_samples:
        raise ValueError(
            f"Recording too short ({n_times} samples) for one {length_s}s clip "
            f"({n_samples} samples)"
        )
    n_clips = 1 + (n_times - n_samples) // step
    windows = np.lib.stride_tricks.sliding_window_view(data, n_samples, axis=1)  # (C, W, T_win)
    windows = windows[:, ::step, :][:, :n_clips, :]  # take exactly n_clips windows
    clips = windows.transpose(1, 0, 2).astype(np.float32)  # (n_clips, C, T)
    if not np.isfinite(clips).all():
        raise ValueError("Non-finite values found after filtering/epoching")
    return clips


def _run_cache_path(
    cache_dir: Path, subject: int, run: int, l_freq: float, h_freq: float, length_s: float
) -> Path:
    return cache_dir / f"S{subject:03d}R{run:02d}_f{l_freq:g}-{h_freq:g}_L{length_s:g}.npz"


def load_run_clips(
    subject: int,
    run: int,
    *,
    l_freq: float,
    h_freq: float,
    length_s: float,
    cache_dir: str | Path,
    data_dir: str | Path | None = None,
    sfreq_expected: float = 160.0,
) -> np.ndarray:
    """Download (if needed), filter and epoch one (subject, run) recording.

    EDF files are downloaded into ``data_dir/eegbci`` (defaults to the
    MNE-managed location if ``data_dir`` is None). Filtered clips are cached
    per (subject, run, filter band, clip length) as .npz in ``cache_dir``.

    Returns:
        float32 array of shape (n_clips, n_channels, n_samples).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _run_cache_path(cache_dir, subject, run, l_freq, h_freq, length_s)
    if cache_path.exists():
        with np.load(cache_path) as z:
            return z["X"]

    download_dir = None if data_dir is None else str(Path(data_dir) / "eegbci")
    if download_dir is not None:
        Path(download_dir).mkdir(parents=True, exist_ok=True)
    paths = eegbci.load_data(
        subject, runs=[run], path=download_dir, update_path=False
    )
    raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")
    eegbci.standardize(raw)  # rename channels to standard_1005 naming + montage
    raw.pick("eeg", verbose="ERROR")
    if abs(raw.info["sfreq"] - sfreq_expected) > 1e-6:
        raise RuntimeError(
            f"Unexpected sampling rate {raw.info['sfreq']} for S{subject:03d}R{run:02d} "
            f"(expected {sfreq_expected} Hz)"
        )
    raw.filter(l_freq, h_freq, method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
    clips = clip_arrays_from_raw(raw, length_s=length_s, overlap_s=0.0)
    np.savez_compressed(cache_path, X=clips)
    return clips


# --------------------------------------------------------------------------- #
# Cue-locked task clips (for the challenge-response defense)
# --------------------------------------------------------------------------- #
def load_run_task_clips(
    subject: int,
    run: int,
    *,
    l_freq: float,
    h_freq: float,
    length_s: float,
    cache_dir: str | Path,
    data_dir: str | Path | None = None,
    cue_offset_s: float = 0.0,
    sfreq_expected: float = 160.0,
) -> np.ndarray:
    """Cut clips locked to the movement-task cue onsets (annotations T1/T2).

    EEGMMIDB marks each trial with annotations: T0 = rest, T1/T2 = the two
    movement conditions of the run's task. Uniformly-cut clips of a motor run
    contain a lot of inter-trial rest (eyes-open), which makes "which task?"
    nearly undecidable from band power; cue-locked clips capture the movement
    itself and are what a challenge-response system would collect.

    Returns:
        float32 array of shape (n_cues, n_channels, n_samples), cached as .npz.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"S{subject:03d}R{run:02d}_cueoff{cue_offset_s:g}_"
        f"f{l_freq:g}-{h_freq:g}_L{length_s:g}.npz"
    )
    if cache_path.exists():
        with np.load(cache_path) as z:
            return z["X"]

    download_dir = None if data_dir is None else str(Path(data_dir) / "eegbci")
    paths = eegbci.load_data(subject, runs=[run], path=download_dir, update_path=False)
    raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    events, code_map = mne.events_from_annotations(raw, verbose="ERROR")
    cue_codes = [c for name, c in code_map.items() if name in ("T1", "T2")]
    ev = events[np.isin(events[:, 2], cue_codes)]
    if len(ev) == 0:
        raise RuntimeError(f"no T1/T2 cue annotations found for S{subject:03d}R{run:02d}")
    raw.pick("eeg", verbose="ERROR")
    raw.filter(l_freq, h_freq, method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
    sfreq = raw.info["sfreq"]
    if abs(sfreq - sfreq_expected) > 1e-6:
        raise RuntimeError(f"unexpected sfreq {sfreq} for S{subject:03d}R{run:02d}")
    data = raw.get_data()  # (C, T)
    n_samples = int(round(length_s * sfreq))
    offset = int(round(cue_offset_s * sfreq))
    n_times = data.shape[1]
    windows = [
        data[:, s + offset : s + offset + n_samples]
        for s in ev[:, 0]
        if 0 <= s + offset and s + offset + n_samples <= n_times
    ]
    clips = np.stack(windows, axis=0).astype(np.float32)  # (n_cues, C, T)
    np.savez_compressed(cache_path, X=clips)
    return clips


# --------------------------------------------------------------------------- #
# Session-based split (the anti-leakage core)
# --------------------------------------------------------------------------- #
def split_by_runs(
    clips: EEGClips,
    train_runs: Sequence[int],
    test_runs: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Split clips into train/test sets by run ID.

    Every clip belongs to exactly one run; a run is assigned to at most one side.
    This guarantees that train and test clips never come from the same recording.

    Args:
        clips: EEGClips with per-clip run IDs.
        train_runs: run numbers used for training/enrollment.
        test_runs: run numbers used for testing/evaluation.

    Returns:
        Dict with keys X_train, y_train, run_ids_train, X_test, y_test,
        run_ids_test. Runs present in neither list are dropped.

    Raises:
        ValueError: if train_runs and test_runs overlap.
    """
    train_set, test_set = set(train_runs), set(test_runs)
    if train_set & test_set:
        raise ValueError(
            f"Data leakage: runs appear in both train and test: {sorted(train_set & test_set)}"
        )
    train_mask = np.isin(clips.run_ids, list(train_set))
    test_mask = np.isin(clips.run_ids, list(test_set))
    return {
        "X_train": clips.X[train_mask],
        "y_train": clips.y[train_mask],
        "run_ids_train": clips.run_ids[train_mask],
        "X_test": clips.X[test_mask],
        "y_test": clips.y[test_mask],
        "run_ids_test": clips.run_ids[test_mask],
    }


# --------------------------------------------------------------------------- #
# Full dataset build with split-level caching
# --------------------------------------------------------------------------- #
def _split_key(cfg: dict[str, Any]) -> str:
    ds, ep, flt = cfg["dataset"], cfg["epochs"], cfg["filter"]
    payload = json.dumps(
        {
            "subjects": sorted(ds["subjects"]),
            "train_runs": sorted(ds["train_runs"]),
            "test_runs": sorted(ds["test_runs"]),
            "l_freq": flt["l_freq"],
            "h_freq": flt["h_freq"],
            "length_s": ep["length_s"],
            "overlap_s": ep["overlap_s"],
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:10]


def build_dataset(cfg: dict[str, Any], use_cache: bool = True) -> Dict[str, np.ndarray]:
    """Build the train/test split defined by the config.

    Downloads any missing (subject, run) recordings, epochs them, labels clips
    by subject index, applies the session-based split, and caches the result.

    Subject class indices are assigned over the *sorted* subject list, so they
    are stable regardless of the order in the config.

    Returns:
        Dict with X_train, y_train, run_ids_train, X_test, y_test, run_ids_test,
        subject_ids_train, subject_ids_test.
    """
    ds, ep, flt = cfg["dataset"], cfg["epochs"], cfg["filter"]
    subjects = sorted(int(s) for s in ds["subjects"])
    train_runs = list(ds["train_runs"])
    test_runs = list(ds["test_runs"])
    if set(train_runs) & set(test_runs):
        raise ValueError(f"Config leakage: train/test runs overlap: {set(train_runs) & set(test_runs)}")

    cache_dir = Path(cfg["paths"]["cache_dir"])
    split_path = cache_dir / f"split_{_split_key(cfg)}.npz"
    if use_cache and split_path.exists():
        with np.load(split_path) as z:
            return {k: z[k] for k in z.files}

    min_clips = int(ep.get("min_clips_per_class", 1))
    keep: List[int] = []
    xs_tr, ys_tr, sid_tr, rid_tr = [], [], [], []
    xs_te, ys_te, sid_te, rid_te = [], [], [], []
    for subject in subjects:
        subj_idx = subjects.index(subject)
        tr_parts, te_parts = [], []
        for run in train_runs:
            X = load_run_clips(
                subject, run, l_freq=flt["l_freq"], h_freq=flt["h_freq"],
                length_s=ep["length_s"], cache_dir=cache_dir,
                data_dir=cfg["paths"].get("data_dir"),
            )
            tr_parts.append(X)
        for run in test_runs:
            X = load_run_clips(
                subject, run, l_freq=flt["l_freq"], h_freq=flt["h_freq"],
                length_s=ep["length_s"], cache_dir=cache_dir,
                data_dir=cfg["paths"].get("data_dir"),
            )
            te_parts.append(X)
        X_tr = np.concatenate(tr_parts, axis=0)
        X_te = np.concatenate(te_parts, axis=0)
        if len(X_tr) < min_clips or len(X_te) < min_clips:
            print(f"[data] WARNING: subject {subject} skipped (too few clips: "
                  f"train={len(X_tr)}, test={len(X_te)})")
            continue
        keep.append(subject)
        xs_tr.append(X_tr)
        ys_tr.append(np.full(len(X_tr), subj_idx, dtype=np.int64))
        sid_tr.append(np.full(len(X_tr), subject, dtype=np.int64))
        rid_tr.append(np.concatenate([np.full(len(X), run, dtype=np.int64)
                                      for X, run in zip(tr_parts, train_runs)]))
        xs_te.append(X_te)
        ys_te.append(np.full(len(X_te), subj_idx, dtype=np.int64))
        sid_te.append(np.full(len(X_te), subject, dtype=np.int64))
        rid_te.append(np.concatenate([np.full(len(X), run, dtype=np.int64)
                                      for X, run in zip(te_parts, test_runs)]))

    clips = EEGClips(
        X=np.concatenate(xs_tr + xs_te, axis=0),
        y=np.concatenate(ys_tr + ys_te, axis=0),
        subject_ids=np.concatenate(sid_tr + sid_te, axis=0),
        run_ids=np.concatenate(rid_tr + rid_te, axis=0),
    )
    split = split_by_runs(clips, train_runs, test_runs)
    split["subject_ids_train"] = clips.subject_ids[np.isin(clips.run_ids, train_runs)]
    split["subject_ids_test"] = clips.subject_ids[np.isin(clips.run_ids, test_runs)]

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, **split)
    return split


def dataset_stats(split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute dataset statistics from a built split (for logging/reporting)."""
    def per_subject_count(y: np.ndarray, sid: np.ndarray) -> Dict[int, int]:
        return {int(s): int(np.sum(sid == s)) for s in np.unique(sid)}

    tr_counts, te_counts = per_subject_count(split["y_train"], split["subject_ids_train"]), \
        per_subject_count(split["y_test"], split["subject_ids_test"])
    subjects = sorted(set(tr_counts) | set(te_counts))
    return {
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_train_clips": int(len(split["y_train"])),
        "n_test_clips": int(len(split["y_test"])),
        "clips_per_subject_train": {str(s): tr_counts.get(s, 0) for s in subjects},
        "clips_per_subject_test": {str(s): te_counts.get(s, 0) for s in subjects},
        "train_clips_min": int(min(tr_counts.values())),
        "train_clips_max": int(max(tr_counts.values())),
        "test_clips_min": int(min(te_counts.values())),
        "test_clips_max": int(max(te_counts.values())),
        "train_runs": sorted(int(r) for r in np.unique(split["run_ids_train"])),
        "test_runs": sorted(int(r) for r in np.unique(split["run_ids_test"])),
        "n_channels": int(split["X_train"].shape[1]),
        "n_samples_per_clip": int(split["X_train"].shape[2]),
        "class_balance_ok": len(set(tr_counts.values())) <= 2 and len(set(te_counts.values())) <= 2,
    }
