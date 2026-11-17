"""Band-power features for EEG clips.

For each clip (n_channels x n_samples) we compute Welch's power spectral
density per channel and average it within standard EEG frequency bands,
then take log10. Feature ordering is channel-major: for channel c the
features are [theta, alpha, beta, gamma] at positions c*B .. c*B+B-1.

No normalisation is done here: scalers must be fitted on TRAIN features only
(see eegauth.train).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import welch

__all__ = ["log_band_power_features", "band_feature_names", "default_bands"]

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def log_band_power_features(
    clips: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]] | None = None,
    nperseg: int | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute per-channel log10 band power for a batch of clips.

    Args:
        clips: float array (n_clips, n_channels, n_samples).
        sfreq: sampling rate in Hz.
        bands: mapping band name -> (low, high) Hz. Defaults to DEFAULT_BANDS.
        nperseg: Welch segment length in samples. Defaults to n_samples
            (one segment per clip: the clip IS the segment).
        batch_size: clips processed per Welch call to bound memory.

    Returns:
        float64 array (n_clips, n_channels * n_bands), channel-major.
    """
    if clips.ndim != 3:
        raise ValueError(f"expected (n, C, T), got shape {clips.shape}")
    bands = bands or DEFAULT_BANDS
    band_list: List[Tuple[str, Tuple[float, float]]] = sorted(
        bands.items(), key=lambda kv: kv[1][0]
    )  # order bands by frequency: theta, alpha, beta, gamma
    n_clips, n_channels, n_samples = clips.shape
    nperseg = nperseg or n_samples
    if nperseg > n_samples:
        raise ValueError(f"nperseg={nperseg} exceeds clip length {n_samples}")

    freqs = None
    out = np.empty((n_clips, n_channels * len(band_list)), dtype=np.float64)
    for start in range(0, n_clips, batch_size):
        batch = clips[start : start + batch_size]
        freqs, psd = welch(batch, fs=sfreq, nperseg=nperseg, axis=-1)  # (n,C,F)
        cols = []
        for _, (lo, hi) in band_list:
            mask = (freqs >= lo) & (freqs <= hi)
            if not mask.any():
                raise ValueError(f"band [{lo}, {hi}] Hz empty at sfreq={sfreq}")
            cols.append(psd[..., mask].mean(axis=-1))
        power = np.stack(cols, axis=-1)  # (n, C, B)
        out[start : start + batch.shape[0]] = (
            np.log10(power).reshape(batch.shape[0], -1)
        )
    return out


def band_feature_names(
    n_channels: int, bands: Dict[str, Tuple[float, float]] | None = None
) -> List[str]:
    """Human-readable feature names, matching the channel-major ordering."""
    bands = bands or DEFAULT_BANDS
    band_list = sorted(bands.items(), key=lambda kv: kv[1][0])
    return [f"ch{c}_{name}" for c in range(n_channels) for name, _ in band_list]
