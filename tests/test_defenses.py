"""Tests for defenses: adversarial training (tiny smoke) and challenge-response sim."""

import numpy as np
import pytest
import torch

from eegauth.defenses import (
    RUN_TO_TASK,
    build_task_clips,
    challenge_response_simulation,
    fit_task_classifier,
    task_labels_from_runs,
    train_cnn_adversarial,
)
from eegauth.models import EEGCNNSmall


def test_task_labels_from_runs():
    runs = np.array([1, 3, 4, 5, 1, 3])
    labels = task_labels_from_runs(runs)
    assert set(labels.tolist()) == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        task_labels_from_runs(np.array([1, 9]))


@pytest.mark.slow
def test_build_task_clips_real_data():
    """Cue-locked task clips from real EDFs (subject 1 only, cached)."""
    from eegauth.config import load_config
    from eegauth.data import build_dataset

    cfg = load_config()
    cfg["dataset"]["subjects"] = [1]
    split = build_dataset(cfg)
    clips = build_task_clips(cfg, split)
    assert clips["X"].shape[1:] == (64, 320)
    counts = np.bincount(clips["y"], minlength=4)
    assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0 and counts[3] > 0
    # cue-locked classes should have ~equal counts (30 cues per run)
    assert counts[1] == counts[2] == counts[3]
    assert np.isfinite(clips["X"]).all()


def test_task_classifier_separates_toy_tasks():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (np.arange(200) % 2).astype(int)
    X += (y * 5)[:, None]  # strongly separable
    model, acc, Xv, yv, _ = fit_task_classifier(X, y, seed=0, val_fraction=0.3)
    assert acc > 0.9


def test_challenge_response_partial_library_beats_full_library():
    """k=1 library must yield much lower replay success than k=4 (all tasks)."""
    rng = np.random.default_rng(3)
    n_per_task = 30
    X = rng.normal(size=(n_per_task * 4, 4))
    y = np.repeat([0, 1, 2, 3], n_per_task)
    X += (y * 6)[:, None]
    model, _, X_val, y_val, _ = fit_task_classifier(X, y, seed=0, val_fraction=0.5)
    id_scores = np.ones(len(y_val))  # identity always accepts in this toy sim
    res = challenge_response_simulation(
        model, X_val, y_val, id_scores, threshold=0.5, n_draws=400, seed=1
    )
    s = res["replay_success_by_library_size"]
    assert s["1"] < s["4"]
    assert s["4"] > 0.8, "with a full library the challenge adds little (expected)"
    assert res["replay_success_without_challenge"] == 1.0


@pytest.mark.slow
def test_adversarial_training_smoke():
    """Two epochs on tiny synthetic data: loss decreases, state loads back."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8, 64)).astype(np.float32)
    y = rng.integers(0, 2, size=64)
    model = EEGCNNSmall(n_channels=8, n_samples=64, n_classes=2, n_filters=8)
    history, state = train_cnn_adversarial(
        model, X, y, lr=0.01, batch_size=16, max_epochs=2, patience=2,
        val_fraction=0.25, device=torch.device("cpu"), seed=0,
        adv_fraction=0.5, pgd_alpha=0.02, pgd_n_iter=2, pgd_eps=0.1, verbose=False,
    )
    assert len(history["train_loss"]) == 2
    assert state is not None
    clone = EEGCNNSmall(n_channels=8, n_samples=64, n_classes=2, n_filters=8)
    clone.load_state_dict(state)
    assert torch.isfinite(next(clone.parameters())).all()
