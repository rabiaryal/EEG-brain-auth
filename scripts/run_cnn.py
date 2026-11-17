#!/usr/bin/env python
"""Train and evaluate the Phase 3 CNN on raw 64x320 clips.

Usage:
    python scripts/run_cnn.py [--config configs/default.yaml]

Outputs:
    models/cnn.pt                      best checkpoint (state dict + meta)
    models/normalizer.npz              per-channel stats (TRAIN only)
    results/cnn.json                   identification + verification metrics
    results/figures/cnn_loss.png       train/val loss curves
    results/figures/roc_cnn.png        verification ROC
    results/comparison.json            baseline vs CNN comparison table
"""

from __future__ import annotations

import argparse
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
from eegauth.metrics import roc_points, verification_metrics
from eegauth.models import EEGCNNSmall, get_device
from eegauth.train import (
    PerChannelNormalizer,
    cnn_softmax_scores,
    genuine_impostor_scores,
    identification_accuracy,
    train_cnn,
)


def evaluate_predictions(pred: np.ndarray, scores: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    acc = float(np.mean(pred == y_test))
    genuine, impostor = genuine_impostor_scores(scores, y_test)
    m = verification_metrics(genuine, impostor)
    fpr, tpr, _ = roc_points(genuine, impostor)
    print(f"[{name}] identification accuracy: {acc:.4f} | EER: {m.eer:.4f} "
          f"| FAR@EER: {m.far_at_eer:.4f} | FRR@EER: {m.frr_at_eer:.4f} | AUC: {m.auc:.4f}")
    return {"name": name, "identification_accuracy": acc, **m.as_dict(),
            "n_genuine_scores": int(genuine.size), "n_impostor_scores": int(impostor.size),
            "_roc": (fpr, tpr)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = set_seeds(cfg)
    device = get_device()
    print(f"[run_cnn] seeds: {seeds} | device: {device}")

    split = build_dataset(cfg)
    n_classes = int(split["y_train"].max()) + 1

    # Per-channel normalisation fitted on TRAIN clips only
    norm = PerChannelNormalizer().fit(split["X_train"])
    min_std = float(norm.std_.min())
    print(f"[run_cnn] normalizer fitted on train only; min channel std: {min_std:.3e} "
          f"({'ok' if min_std > 1e-6 else 'WARNING: near-flat channel'})")
    X_tr = norm.transform(split["X_train"])
    X_te = norm.transform(split["X_test"])

    cnn_cfg = cfg["models"]["cnn"]
    torch = __import__("torch")
    model = EEGCNNSmall(
        n_channels=X_tr.shape[1], n_samples=X_tr.shape[2],
        n_classes=n_classes, n_filters=cnn_cfg["n_filters"], dropout=cnn_cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run_cnn] model params: {n_params}")

    t0 = time.time()
    history, _ = train_cnn(
        model, X_tr, split["y_train"],
        lr=cnn_cfg["lr"], batch_size=cnn_cfg["batch_size"],
        max_epochs=cnn_cfg["max_epochs"], patience=cnn_cfg["patience"],
        val_fraction=cnn_cfg["val_fraction"], device=device, seed=seeds["torch"],
    )
    train_s = time.time() - t0
    print(f"[run_cnn] training took {train_s:.1f}s, {len(history['train_loss'])} epochs")

    # Save checkpoint + normalizer for later phases (attacks, API)
    models_dir = Path(cfg["paths"]["models_dir"])
    torch.save({
        "state_dict": model.state_dict(),
        "meta": {"n_channels": int(X_tr.shape[1]), "n_samples": int(X_tr.shape[2]),
                 "n_classes": int(n_classes), "n_filters": cnn_cfg["n_filters"],
                 "dropout": cnn_cfg["dropout"]},
    }, models_dir / "cnn.pt")
    norm.save(models_dir / "normalizer.npz")

    # Loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="val loss (train-run split)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("CNN training curves")
    ax.legend()
    loss_path = Path(cfg["paths"]["figures_dir"]) / "cnn_loss.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")

    # Evaluation on test runs
    model.eval()
    with torch.no_grad():
        probs = cnn_softmax_scores(model, X_te, device=device)
    res = evaluate_predictions(probs.argmax(1), probs, split["y_test"], "cnn")
    res["n_params"] = int(n_params)
    res["train_seconds"] = round(train_s, 1)
    res["n_epochs"] = len(history["train_loss"])
    res["best_val_loss"] = min(history["val_loss"])

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fpr, tpr = res.pop("_roc")
    ax2.plot(fpr, tpr, label=f"CNN (EER={res['eer']:.3f}, AUC={res['auc']:.3f})")
    ax2.plot([0, 1], [1, 0], "k--", lw=1, label="chance")
    ax2.set_xlabel("FAR (FPR)")
    ax2.set_ylabel("1 - FRR (TPR)")
    ax2.set_title("Verification ROC — CNN (20 subjects, test runs)")
    ax2.legend(loc="upper left", fontsize=9)
    roc_path = Path(cfg["paths"]["figures_dir"]) / "roc_cnn.png"
    fig2.savefig(roc_path, dpi=150, bbox_inches="tight")
    print(f"[run_cnn] figures -> {loss_path.name}, {roc_path.name}")

    results = {"seeds": seeds, "device": str(device), **{k: v for k, v in res.items()}}
    with open(Path(cfg["paths"]["results_dir"]) / "cnn.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Comparison table: merge with baseline results if present
    base_path = Path(cfg["paths"]["results_dir"]) / "baseline.json"
    table = {}
    if base_path.exists():
        with open(base_path, encoding="utf-8") as f:
            base = json.load(f)
        for name, r in base["models"].items():
            table[f"baseline_{name}"] = {
                "identification_accuracy": r["identification_accuracy"],
                "eer": r["eer"], "auc": r["auc"],
            }
    table["cnn"] = {
        "identification_accuracy": res["identification_accuracy"],
        "eer": res["eer"], "auc": res["auc"],
    }
    with open(Path(cfg["paths"]["results_dir"]) / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    print("\n| Model | Identification acc | EER | AUC |")
    print("|---|---|---|---|")
    for name, r in table.items():
        print(f"| {name} | {r['identification_accuracy']:.2%} | {r['eer']:.2%} | {r['auc']:.4f} |")


if __name__ == "__main__":
    main()
