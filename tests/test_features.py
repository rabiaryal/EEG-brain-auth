"""Tests for band-power features (synthetic signals with known spectra)."""

import numpy as np
import pytest

from eegauth.features import band_feature_names, log_band_power_features


def test_output_shape_and_ordering():
    rng = np.random.default_rng(0)
    clips = rng.standard_normal((5, 64, 320)) * 1e-5
    feats = log_band_power_features(clips, sfreq=160.0)
    assert feats.shape == (5, 64 * 4)  # 4 bands, channel-major


def test_pure_sine_dominates_its_band():
    """A 10 Hz sine must have its alpha feature above its theta/beta features."""
    sfreq = 160.0
    t = np.arange(320) / sfreq
    # 10 Hz sine on channel 0, white noise elsewhere
    clip = np.zeros((1, 2, 320))
    clip[0, 0] = 1e-5 * np.sin(2 * np.pi * 10.0 * t)
    clip[0, 1] = 1e-6 * np.sin(2 * np.pi * 10.0 * t)  # quieter control channel
    feats = log_band_power_features(clip, sfreq=sfreq)
    # channel 0: [theta, alpha, beta, gamma]
    th, al, be, ga = feats[0, 0:4]
    assert al > th
    assert al > be
    # channel 1 (same 10 Hz but 20 dB quieter) must have lower alpha power
    assert feats[0, 4 + 1] < al


def test_log10_transform():
    """Doubled amplitude = +6 dB = +0.602 in log10 power."""
    sfreq = 160.0
    t = np.arange(320) / sfreq
    clip = np.zeros((2, 1, 320))
    clip[0, 0] = 1e-5 * np.sin(2 * np.pi * 10.0 * t)
    clip[1, 0] = 2e-5 * np.sin(2 * np.pi * 10.0 * t)
    feats = log_band_power_features(clip, sfreq=sfreq)
    diff = feats[1, 1] - feats[0, 1]  # alpha band
    assert diff == pytest.approx(np.log10(4.0), abs=0.05)


def test_batching_matches_single_call():
    rng = np.random.default_rng(1)
    clips = rng.standard_normal((40, 8, 320))
    a = log_band_power_features(clips, sfreq=160.0, batch_size=7)
    b = log_band_power_features(clips, sfreq=160.0, batch_size=64)
    assert np.allclose(a, b)


def test_feature_names():
    names = band_feature_names(2)
    assert names[:4] == ["ch0_theta", "ch0_alpha", "ch0_beta", "ch0_gamma"]
    assert len(names) == 8


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        log_band_power_features(np.zeros((5, 64)), sfreq=160.0)
    with pytest.raises(ValueError):
        log_band_power_features(np.zeros((5, 64, 100)), sfreq=160.0, nperseg=200)
