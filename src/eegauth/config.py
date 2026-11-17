"""Configuration loading and reproducibility helpers.

The project uses a single YAML config (configs/default.yaml). This module loads it
into a plain dict, resolves relative paths against the repository root, and provides
seed fixing so every entry point produces reproducible runs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML config into a dict.

    Relative paths under ``paths:`` are resolved to absolute paths against the
    repository root so scripts work from any working directory.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str) and not Path(value).is_absolute():
            cfg["paths"][key] = str((REPO_ROOT / value).resolve())
    return cfg


def repo_root() -> Path:
    """Return the repository root directory."""
    return REPO_ROOT


def set_seeds(cfg: dict[str, Any]) -> dict[str, int]:
    """Fix random seeds for python, numpy and torch (if installed).

    Returns the seeds that were set so callers can log them.
    """
    seeds = cfg.get("seeds", {})
    py_seed = int(seeds.get("python", seeds.get("numpy", 42)))
    np_seed = int(seeds.get("numpy", 42))
    random.seed(py_seed)
    np.random.seed(np_seed)
    os.environ["PYTHONHASHSEED"] = str(py_seed)
    try:
        import torch

        torch.manual_seed(int(seeds.get("torch", 42)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seeds.get("torch", 42)))
    except ImportError:
        pass
    return {
        "python": py_seed,
        "numpy": np_seed,
        **({"torch": int(seeds["torch"])} if "torch" in seeds else {}),
    }
