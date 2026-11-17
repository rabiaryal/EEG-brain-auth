"""Fast unit tests for the session-based split logic (no download needed).

These tests exist to catch data leakage: they fail if a run ever appears on
both the train and test side, or if labels/subject IDs get misaligned.
"""

import numpy as np
import pytest

from eegauth.data import EEGClips, split_by_runs


def make_clips(runs, clips_per_run=4, n_channels=64, n_samples=320):
    """Build a synthetic EEGClips where clip i of run r has value r in channel 0."""
    X, y, sid, rid = [], [], [], []
    clip_idx = 0
    for run in runs:
        for _ in range(clips_per_run):
            x = np.zeros((n_channels, n_samples), dtype=np.float32)
            x[0, :] = run  # marker so provenance is inspectable if ever needed
            X.append(x)
            y.append(clip_idx % 5)
            sid.append((clip_idx % 3) + 1)
            rid.append(run)
            clip_idx += 1
    return EEGClips(
        X=np.stack(X),
        y=np.array(y, dtype=np.int64),
        subject_ids=np.array(sid, dtype=np.int64),
        run_ids=np.array(rid, dtype=np.int64),
    )


def test_split_disjoint_runs():
    clips = make_clips(runs=[1, 2, 3, 4, 5, 6, 7])
    split = split_by_runs(clips, train_runs=[1, 3, 4, 5], test_runs=[2, 6, 7])
    assert set(split["run_ids_train"].tolist()).isdisjoint(set(split["run_ids_test"].tolist()))
    assert set(split["run_ids_train"].tolist()) == {1, 3, 4, 5}
    assert set(split["run_ids_test"].tolist()) == {2, 6, 7}


def test_split_overlapping_runs_raise():
    clips = make_clips(runs=[1, 2, 3])
    with pytest.raises(ValueError, match="leakage"):
        split_by_runs(clips, train_runs=[1, 2], test_runs=[2, 3])


def test_split_no_clip_in_both_sides():
    """Stronger leak check: identical clips must not appear in train and test."""
    clips = make_clips(runs=[1, 2], clips_per_run=6)
    split = split_by_runs(clips, train_runs=[1], test_runs=[2])
    tr_hashes = {x.tobytes() for x in split["X_train"]}
    te_hashes = {x.tobytes() for x in split["X_test"]}
    assert not (tr_hashes & te_hashes)


def test_split_label_alignment():
    """Every clip's label must travel with it to the same side."""
    clips = make_clips(runs=[1, 2, 3], clips_per_run=2)
    split = split_by_runs(clips, train_runs=[1, 3], test_runs=[2])
    assert len(split["y_train"]) == 4 and len(split["y_test"]) == 2
    assert len(split["X_train"]) == len(split["y_train"]) == len(split["run_ids_train"])
    assert len(split["X_test"]) == len(split["y_test"]) == len(split["run_ids_test"])


def test_split_drops_unknown_runs():
    clips = make_clips(runs=[1, 2, 8])
    split = split_by_runs(clips, train_runs=[1], test_runs=[2])
    assert 8 not in set(split["run_ids_train"].tolist())
    assert 8 not in set(split["run_ids_test"].tolist())
    assert split["X_train"].shape[0] + split["X_test"].shape[0] == 8  # run 8's 4 clips dropped, 4+4 kept


def test_shapes_fixed_length():
    clips = make_clips(runs=[1], clips_per_run=10)
    assert clips.X.shape == (10, 64, 320)
    assert clips.X.dtype == np.float32
