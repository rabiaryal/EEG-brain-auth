"""Slow integration tests using REAL downloaded EEGMMIDB data.

Marked ``slow``: run with ``pytest -m slow``. Uses 2 subjects to keep the
download small. Verifies shapes, run disjointness (leakage), and label
consistency on the actual pipeline output.
"""

import numpy as np
import pytest

from eegauth.config import load_config, set_seeds
from eegauth.data import build_dataset, dataset_stats


@pytest.fixture(scope="module")
def small_split():
    cfg = load_config()
    cfg["dataset"]["subjects"] = [1, 2]  # keep the download small
    set_seeds(cfg)
    return build_dataset(cfg)


@pytest.mark.slow
def test_shapes(small_split):
    X = small_split["X_train"]
    assert X.ndim == 3
    assert X.shape[1] == 64, f"expected 64 channels, got {X.shape[1]}"
    assert X.shape[2] == 320, f"expected 320 samples (2 s @ 160 Hz), got {X.shape[2]}"
    assert X.dtype == np.float32
    assert small_split["X_test"].shape[1:] == X.shape[1:]


@pytest.mark.slow
def test_no_run_leakage(small_split):
    """THE leakage test: no run may appear on both sides of the split."""
    tr = set(small_split["run_ids_train"].tolist())
    te = set(small_split["run_ids_test"].tolist())
    assert tr.isdisjoint(te), f"leakage: runs {tr & te} in both train and test"


@pytest.mark.slow
def test_labels_match_subjects(small_split):
    """Label <-> subject mapping must be a consistent bijection."""
    sid_tr, y_tr = small_split["subject_ids_train"], small_split["y_train"]
    sid_te, y_te = small_split["subject_ids_test"], small_split["y_test"]
    map_train = {int(s): int(y) for s, y in zip(np.unique(sid_tr),
                                                [np.unique(y_tr[sid_tr == s])[0] for s in np.unique(sid_tr)])}
    map_test = {int(s): int(y) for s, y in zip(np.unique(sid_te),
                                               [np.unique(y_te[sid_te == s])[0] for s in np.unique(sid_te)])}
    assert map_train == map_test
    assert set(map_train.values()) == set(range(len(map_train)))


@pytest.mark.slow
def test_clip_values_finite_and_plausible(small_split):
    X = np.concatenate([small_split["X_train"], small_split["X_test"]])
    assert np.isfinite(X).all()
    # EEGMMIDB scalp EEG after 1-40 Hz filtering: microvolt scale, not volts.
    rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=(1, 2)))
    assert (rms > 1e-8).all(), "clip RMS suspiciously small (dead data?)"
    assert (rms < 1e-2).all(), "clip RMS suspiciously large (units wrong?)"


@pytest.mark.slow
def test_stats_consistent(small_split):
    stats = dataset_stats(small_split)
    assert stats["n_subjects"] == 2
    assert stats["n_train_clips"] + stats["n_test_clips"] > 0
    assert stats["train_runs"] == [1, 3, 4, 5]
    assert stats["test_runs"] == [2, 6, 7]
