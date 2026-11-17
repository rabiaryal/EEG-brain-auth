#!/usr/bin/env python
"""Train and evaluate the Phase 2 baselines (LogReg + SVM) on band-power features.

Usage:
    python scripts/run_baseline.py [--config configs/default.yaml] [--skip-svm]

Outputs:
    results/baseline.json              metrics for both baselines
    results/figures/roc_baseline.png   ROC curves (genuine vs impostor)
    data/cache/features_<key>.npz      cached feature matrices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eegauth.config import load_config, set_seeds
from eegauth.data import build_dataset
from eegauth.features import log_band_power_features
from eegauth.metrics import roc_points, verification_metrics
from eegauth.train import fit_baseline, genuine_impostor_scores, identification_accuracy, score_matrix


def feature_cache_path(cfg: dict, split: dict) -> Path:
    payload = json.dumps(
        {
            "bands": cfg["features"]["bands"],
            "nperseg": cfg["features"]["welch_nperseg"],
            "n_train": int(split["X_train"].shape[0]),
            "n_test": int(split["X_test"].shape[0]),
            "y_train_sum": int(split["y_train"].sum()),
            "y_test_sum": int(split["y_test"].sum()),
        },
        sort_keys=True,
    )
    key = hashlib.md5(payload.encode()).hexdigest()[:10]
    return Path(cfg["paths"]["cache_dir"]) / f"features_{key}.npz"


def get_features(cfg: dict, split: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute (or load cached) band-power features for train and test clips."""
    cache = feature_cache_path(cfg, split)
    if cache.exists():
        with np.load(cache) as z:
            return z["Xf_train"], z["Xf_test"]
    t0 = time.time()
    bands = {k: tuple(v) for k, v in cfg["features"]["bands"].items()}
    Xf_train = log_band_power_features(
        split["X_train"], sfreq=cfg["dataset"]["expected_sfreq"],
        bands=bands, nperseg=cfg["features"]["welch_nperseg"],
    )
    Xf_test = log_band_power_features(
        split["X_test"], sfreq=cfg["dataset"]["expected_sfreq"],
        bands=bands, nperseg=cfg["features"]["welch_nperseg"],
    )
    print(f"[features] computed in {time.time() - t0:.1f}s -> cache {cache.name}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, Xf_train=Xf_train, Xf_test=Xf_test)
    return Xf_train, Xf_test


def evaluate_model(name: str, model, split: dict, Xf_test: np.ndarray) -> dict:
    n_classes = int(split["y_train"].max()) + 1
    acc = identification_accuracy(model, Xf_test, split["y_test"])
    scores = score_matrix(model, Xf_test, n_classes)
    genuine, impostor = genuine_impostor_scores(scores, split["y_test"])
    m = verification_metrics(genuine, impostor)
    fpr, tpr, _ = roc_points(genuine, impostor)
    print(
        f"[{name}] identification accuracy: {acc:.4f} | EER: {m.eer:.4f} "
        f"| FAR@EER: {m.far_at_eer:.4f} | FRR@EER: {m.frr_at_eer:.4f} | AUC: {m.auc:.4f}"
    )
    return {
        "name": name,
        "identification_accuracy": acc,
        **m.as_dict(),
        "n_genuine_scores": int(genuine.size),
        "n_impostor_scores": int(impostor.size),
        "_roc": (fpr, tpr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-svm", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = set_seeds(cfg)
    print(f"[run_baseline] seeds: {seeds}")

    split = build_dataset(cfg)
    Xf_train, Xf_test = get_features(cfg, split)
    n_features = Xf_train.shape[1]
    print(f"[run_baseline] features: {Xf_train.shape} -> {Xf_test.shape} ({n_features} dims)")

    results: dict = {"seeds": seeds, "n_features": int(n_features), "models": {}}
    fig, ax = plt.subplots(figsize=(6, 5))

    bl_cfg = cfg["models"]["baseline"]
    for name, kind, kwargs in [
        ("logreg", "logreg", {"C": bl_cfg["logreg_C"]}),
        ("svm", "svm", {"C": bl_cfg["svm_C"], "gamma": bl_cfg["svm_gamma"]}),
    ]:
        if name == "svm" and args.skip_svm:
            continue
        t0 = time.time()
        model = fit_baseline(Xf_train, split["y_train"], kind=kind, **kwargs)
        fit_s = time.time() - t0
        res = evaluate_model(name, model, split, Xf_test)
        res["fit_seconds"] = round(fit_s, 1)
        results["models"][name] = res
        fpr, tpr = res.pop("_roc")
        ax.plot(fpr, tpr, label=f"{name} (EER={res['eer']:.3f}, AUC={res['auc']:.3f})")

    ax.plot([0, 1], [1, 0], "k--", lw=1, label="chance (anti-diagonal)")
    ax.set_xlabel("FAR (FPR)")
    ax.set_ylabel("1 - FRR (TPR)")
    ax.set_title("Verification ROC — baseline models (20 subjects, test runs)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig_path = Path(cfg["paths"]["figures_dir"]) / "roc_baseline.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[run_baseline] ROC figure -> {fig_path}")

    out = Path(cfg["paths"]["results_dir"]) / "baseline.json"
    serializable = {k: v for k, v in results.items() if k != "models"} | {
        "models": {n: {k: v for k, v in r.items() if not k.startswith("_")}
                   for n, r in results["models"].items()}
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"[run_baseline] metrics -> {out}")


if __name__ == "__main__":
    main()
