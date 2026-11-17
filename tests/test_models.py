"""Tests for the CNN model and per-channel normalizer (CPU, tiny inputs)."""

import numpy as np
import pytest
import torch

from eegauth.models import EEGCNNSmall
from eegauth.train import PerChannelNormalizer


def test_cnn_forward_shape():
    torch.manual_seed(0)
    model = EEGCNNSmall(n_channels=64, n_samples=320, n_classes=20)
    x = torch.randn(4, 64, 320)
    logits = model(x)
    assert logits.shape == (4, 20)
    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 < n_params < 500_000, f"unexpected param count {n_params}"


def test_cnn_forward_other_shapes():
    model = EEGCNNSmall(n_channels=64, n_samples=320, n_classes=7, n_filters=16)
    assert model(torch.randn(2, 64, 320)).shape == (2, 7)


def test_normalizer_fit_transform_statistics():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=3.0, scale=2.0, size=(64, 4, 50)).astype(np.float64)
    # make channel 2 have a different scale
    X[:, 2, :] *= 100.0
    norm = PerChannelNormalizer().fit(X)
    Xt = norm.transform(X)
    # after normalisation, each channel is ~unit variance / ~zero mean (float32)
    assert np.allclose(Xt.mean(axis=(0, 2)), 0.0, atol=1e-6)
    assert np.allclose(Xt.std(axis=(0, 2)), 1.0, atol=1e-3)


def test_normalizer_uses_train_stats_for_test():
    rng = np.random.default_rng(1)
    X_train = rng.normal(5.0, 1.0, size=(32, 3, 20))
    X_test = rng.normal(5.0, 1.0, size=(8, 3, 20)) + 50.0  # shifted!
    norm = PerChannelNormalizer().fit(X_train)
    Xt = norm.transform(X_test)
    # with train stats the output stays near the 50-sigma offset; with test
    # stats it would be centred at ~0. Allow generous sampling noise (n=160).
    means = Xt.mean(axis=(0, 2))
    assert np.all(np.abs(means) > 30.0), f"test data was normalised with TEST stats: {means}"


def test_normalizer_zero_variance_channel():
    """Dead channel (std=0) must not produce NaN/inf (C4-style reference)."""
    X = np.ones((10, 4, 20)) * 1e-7
    X[:, 0, :] = 0.0  # flat channel
    norm = PerChannelNormalizer().fit(X)
    Xt = norm.transform(X)
    assert np.isfinite(Xt).all()


def test_normalizer_save_load_roundtrip():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(16, 5, 30))
    norm = PerChannelNormalizer().fit(X)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "norm.npz"
        norm.save(p)
        norm2 = PerChannelNormalizer.load(p)
    assert np.allclose(norm.mean_, norm2.mean_)
    assert np.allclose(norm.std_, norm2.std_)
    assert np.allclose(norm.transform(X), norm2.transform(X))


def test_normalizer_requires_fit():
    with pytest.raises(RuntimeError):
        PerChannelNormalizer().transform(np.zeros((2, 3, 4)))
