"""Unit tests for verification metrics on toy examples with KNOWN answers."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from eegauth.metrics import eer_from_roc, roc_points, verification_metrics


def test_perfectly_separated_scores_have_zero_eer():
    genuine = np.array([0.61, 0.7, 0.75, 0.83, 0.9, 0.95])
    impostor = np.array([0.01, 0.05, 0.12, 0.2, 0.31, 0.4])
    m = verification_metrics(genuine, impostor)
    assert m.eer == pytest.approx(0.0, abs=1e-6)
    assert m.far_at_eer == pytest.approx(0.0, abs=1e-6)
    assert m.frr_at_eer == pytest.approx(0.0, abs=1e-6)
    assert m.auc == pytest.approx(1.0)


def test_identical_distributions_have_half_eer():
    """Genuine and impostor scores drawn from the same multiset -> EER = 0.5."""
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    m = verification_metrics(scores, scores.copy())
    assert m.eer == pytest.approx(0.5, abs=1e-6)
    assert m.auc == pytest.approx(0.5)


def test_reversed_scores_are_worse_than_chance():
    genuine = np.array([0.1, 0.15, 0.2])
    impostor = np.array([0.5, 0.6, 0.7])
    m = verification_metrics(genuine, impostor)
    assert m.eer >= 0.5
    assert m.auc < 0.5


def test_far_equals_frr_at_eer_threshold():
    rng = np.random.default_rng(7)
    genuine = rng.normal(0.7, 0.1, size=500)
    impostor = rng.normal(0.4, 0.1, size=5000)
    m = verification_metrics(genuine, impostor)
    assert m.far_at_eer == pytest.approx(m.frr_at_eer, abs=0.05)
    assert 0.0 < m.eer < 0.5


def test_eer_between_roc_points_interpolates():
    """EER must lie between adjacent ROC points that straddle the diagonal."""
    genuine = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    impostor = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    fpr, tpr, _ = roc_points(genuine, impostor)
    eer, _ = eer_from_roc(fpr, tpr)
    # crossing segment is (0.4, 0.4) -> (0.6, 0.4)
    assert 0.4 <= eer <= 0.6


def test_roc_curve_monotone():
    rng = np.random.default_rng(3)
    genuine = rng.normal(0.6, 0.15, size=200)
    impostor = rng.normal(0.45, 0.15, size=2000)
    fpr, tpr, thr = roc_points(genuine, impostor)
    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(tpr) >= -1e-12)
    assert np.all(np.diff(thr) <= 1e-12)  # descending


def test_auc_matches_sklearn():
    rng = np.random.default_rng(11)
    genuine = rng.normal(0.7, 0.1, size=300)
    impostor = rng.normal(0.5, 0.1, size=700)
    m = verification_metrics(genuine, impostor)
    y = np.r_[np.ones(300), np.zeros(700)]
    s = np.r_[genuine, impostor]
    assert m.auc == pytest.approx(roc_auc_score(y, s))
