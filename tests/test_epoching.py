"""Fast unit tests for clip epoching geometry (synthetic raw, no download)."""

import numpy as np
import pytest
import mne

from eegauth.data import clip_arrays_from_raw


def make_raw(n_channels=64, sfreq=160.0, duration_s=10.0):
    n_times = int(sfreq * duration_s)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_channels, n_times)).astype(np.float64)
    info = mne.create_info([f"EEG{i:03d}" for i in range(n_channels)], sfreq, "eeg")
    return mne.io.RawArray(data, info)


def test_exact_number_of_nonoverlapping_clips():
    raw = make_raw(duration_s=10.0)  # 1600 samples
    clips = clip_arrays_from_raw(raw, length_s=2.0, overlap_s=0.0)
    assert clips.shape == (5, 64, 320)  # 5 disjoint 2 s clips, remainder dropped


def test_clip_values_are_correct_slices():
    raw = make_raw(n_channels=2, sfreq=160.0, duration_s=4.0)
    raw._data[0, :160] = 1.0
    clips = clip_arrays_from_raw(raw, length_s=1.0, overlap_s=0.0)
    assert clips.shape == (4, 2, 160)
    assert np.all(clips[0, 0] == 1.0)
    assert np.all(clips[1, 0] != 1.0)


def test_overlap_clips():
    raw = make_raw(n_channels=1, sfreq=160.0, duration_s=3.0)
    raw._data[0, :] = np.arange(480)
    clips = clip_arrays_from_raw(raw, length_s=2.0, overlap_s=1.0)
    # step=1s=160 samples: starts at 0,160 -> clips [[0..319],[160..479]]
    assert clips.shape == (2, 1, 320)
    assert clips[0, 0, 0] == 0 and clips[1, 0, 0] == 160


def test_too_short_raises():
    raw = make_raw(duration_s=1.0)
    with pytest.raises(ValueError, match="too short"):
        clip_arrays_from_raw(raw, length_s=2.0)


def test_output_float32():
    raw = make_raw(duration_s=2.0)
    clips = clip_arrays_from_raw(raw, length_s=2.0)
    assert clips.dtype == np.float32
