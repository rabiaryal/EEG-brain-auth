"""Clip ingestion for the API: .npy arrays or .edf recordings.

Every uploaded file is converted to a float32 array of shape
(n_clips, 64, 320) — 2 s clips at 160 Hz — or an HTTP 400 is raised.
"""

from __future__ import annotations

import io
import re
from typing import Tuple

import numpy as np

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
EXPECTED_SHAPE = (64, 320)  # 64 channels x 320 samples (2 s @ 160 Hz)
MIN_CHANNEL_STD = 1e-8


def validate_user_id(user_id: str) -> str:
    """Reject anything that could escape the per-user storage directory."""
    if not USER_ID_RE.match(user_id or ""):
        raise ValueError(
            "user_id must match [A-Za-z0-9_-]{1,64} (no slashes or dots allowed)"
        )
    return user_id


def clips_from_npy(data: bytes) -> np.ndarray:
    """Parse a .npy upload into a validated (n, 64, 320) float32 array."""
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse .npy upload: {exc}") from exc
    return _validate_clip_array(arr)


def _validate_clip_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"clips must have 3 dimensions (n, 64, 320); got shape {arr.shape}")
    if arr.shape[1:] != EXPECTED_SHAPE:
        raise ValueError(
            f"each clip must be 64 channels x 320 samples (2 s at 160 Hz); "
            f"got {arr.shape[1:]}. Resample/re-chunk your recording to 160 Hz / 2 s."
        )
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("clips contain NaN or infinite values")
    rms = np.sqrt((arr.astype(np.float64) ** 2).mean(axis=(1, 2)))
    if (rms < MIN_CHANNEL_STD).any():
        raise ValueError("one or more clips are all-zero (dead input)")
    return arr


def clips_from_edf(data: bytes) -> np.ndarray:
    """Parse an .edf upload (single or multiple recordings concatenated).

    Requires the recording to be 160 Hz with >= 64 EEG channels; the montage is
    standardised and filtered exactly like the training pipeline.
    """
    import mne

    import os
    import tempfile

    # MNE's EDF reader needs a real file path.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "upload.edf")
        with open(path, "wb") as fh:
            fh.write(data)
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            mne.datasets.eegbci.standardize(raw)
            raw.pick("eeg", verbose="ERROR")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"could not parse .edf upload: {exc}") from exc
    if abs(raw.info["sfreq"] - 160.0) > 1e-6:
        raise ValueError(f"EDF must be sampled at 160 Hz; got {raw.info['sfreq']} Hz")
    if len(raw.info["ch_names"]) < 64:
        raise ValueError(f"EDF must have >= 64 EEG channels; got {len(raw.info['ch_names'])}")
    raw.filter(1.0, 40.0, method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
    data64 = raw.get_data()[:64]  # (C, T), keep the canonical first 64 channels
    n_samples = 320
    n_clips = data64.shape[1] // n_samples
    if n_clips == 0:
        raise ValueError("EDF too short for a single 2 s clip")
    clips = data64[:, : n_clips * n_samples].reshape(64, n_clips, n_samples)
    return _validate_clip_array(clips.transpose(1, 0, 2))


def read_clips_upload(filename: str, data: bytes) -> np.ndarray:
    """Dispatch on file extension (.npy or .edf)."""
    name = (filename or "").lower()
    if name.endswith(".npy"):
        return clips_from_npy(data)
    if name.endswith(".edf"):
        return clips_from_edf(data)
    raise ValueError("unsupported file type: upload a .npy or .edf file")
