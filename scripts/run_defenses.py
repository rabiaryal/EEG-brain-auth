#!/usr/bin/env python
"""Phase 5: defenses — adversarial training + (simulated) challenge-response.

Usage:
    python scripts/run_defenses.py [--config configs/default.yaml]

Outputs:
    models/cnn_robust.pt                       adversarially trained CNN
    results/defenses.json                      before/after + clean cost + CR sim
    results/figures/cnn_robust_loss.png        robust training curves
    results/figures/epsilon_before_after.png   attack curves, clean vs robust
    results/figures/challenge_response.png     replay success vs library size
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eegauth.attacks import fgsm_linf, pgd_linf, sample_attack_pairs, targeted_attack_eval
from eegauth.config import load_config, set_seeds
from eegauth.data import build_dataset
from eegauth.defenses import (
    challenge_response_simulation,
    fit_task_classifier,
    build_task_clips,
    train_cnn_adversarial,
)
from eegauth.features import log_band_power_features
from eegauth.metrics import verification_metrics
from eegauth.models import EEGCNNSmall, get_device
from eegauth.train import PerChannelNormalizer, cnn_softmax_scores, genuine_impostor_scores


def attack_curves(model, X_imp, victims, threshold, epsilons, pgd_cfg, device):
    out = {"fgsm": [], "pgd": []}
    for eps in epsilons:
        for name, fn in [
            ("fgsm", lambda xb, yb: fgsm_linf(model, xb, yb, eps=eps)),
            ("pgd", lambda xb, yb: pgd_linf(model, xb, yb, eps=eps,
                                            alpha=pgd_cfg["alpha"], n_iter=pgd_cfg["n_iter"])),
        ]:
            r = targeted_attack_eval(model, X_imp, victims, threshold, fn, device)
            out[name].append({"epsilon": eps, **r})
    return out


def clean_eval(model, X_te, y_te, device, threshold):
    probs = cnn_softmax_scores(model, X_te, device=device)
    genuine, impostor = genuine_impostor_scores(probs, y_te)
    m = verification_metrics(genuine, impostor)
    acc = float(np.mean(probs.argmax(1) == y_te))
    far = float(np.mean(impostor >= threshold))
    return {"identification_accuracy": acc, "eer": m.eer, "auc": m.auc,
            "impostor_far_at_fixed_threshold": far}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = set_seeds(cfg)
    device = get_device()
    atk_cfg = cfg["attacks"]
    d_cfg = cfg["defenses"]["adv_training"]
    print(f"[run_defenses] seeds {seeds} | device {device}")

    split = build_dataset(cfg)
    norm = PerChannelNormalizer.load(Path(cfg["paths"]["models_dir"]) / "normalizer.npz")
    X_tr = norm.transform(split["X_train"])
    X_te = norm.transform(split["X_test"])
    y_tr, y_te = split["y_train"], split["y_test"]
    n_classes = int(y_tr.max()) + 1

    with open(Path(cfg["paths"]["results_dir"]) / "cnn.json", encoding="utf-8") as f:
        cnn_res = json.load(f)
    threshold = float(cnn_res["eer_threshold"])
    with open(Path(cfg["paths"]["results_dir"]) / "attacks.json", encoding="utf-8") as f:
        before = json.load(f)

    epsilons = list(atk_cfg["epsilons"])
    pgd_cfg = atk_cfg["pgd"]
    X_imp, victims, _ = sample_attack_pairs(X_te, y_te, n_pairs=int(atk_cfg["max_attack_pairs"]),
                                            seed=seeds["numpy"])

    # ---- 5.1 Adversarial training ----
    t0 = time.time()
    robust = EEGCNNSmall(n_channels=X_tr.shape[1], n_samples=X_tr.shape[2], n_classes=n_classes,
                         n_filters=cfg["models"]["cnn"]["n_filters"],
                         dropout=cfg["models"]["cnn"]["dropout"]).to(device)
    history, best_state = train_cnn_adversarial(
        robust, X_tr, y_tr,
        lr=cfg["models"]["cnn"]["lr"], batch_size=cfg["models"]["cnn"]["batch_size"],
        max_epochs=cfg["models"]["cnn"]["max_epochs"], patience=cfg["models"]["cnn"]["patience"],
        val_fraction=cfg["models"]["cnn"]["val_fraction"], device=device, seed=seeds["torch"],
        adv_fraction=d_cfg["adv_fraction"], pgd_alpha=d_cfg["pgd_alpha"],
        pgd_n_iter=d_cfg["pgd_n_iter"], pgd_eps=d_cfg["pgd_eps"],
    )
    train_s = time.time() - t0
    print(f"[run_defenses] adversarial training took {train_s:.1f}s, "
          f"{len(history['train_loss'])} epochs")
    torch.save({"state_dict": best_state,
                "meta": {"n_channels": int(X_tr.shape[1]), "n_samples": int(X_tr.shape[2]),
                         "n_classes": int(n_classes), "n_filters": cfg["models"]["cnn"]["n_filters"],
                         "dropout": cfg["models"]["cnn"]["dropout"]}},
               Path(cfg["paths"]["models_dir"]) / "cnn_robust.pt")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train loss (adv-mixed)")
    ax.plot(history["val_loss"], label="val loss (clean)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title("Adversarial training curves")
    loss_path = Path(cfg["paths"]["figures_dir"]) / "cnn_robust_loss.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")

    clean_before = clean_eval_from_json(cnn_res, threshold)
    clean_after = clean_eval(robust, X_te, y_te, device, threshold)
    print(f"[5.1 clean cost] before: acc {clean_before['identification_accuracy']:.4f} "
          f"EER {clean_before['eer']:.4f} | after: acc {clean_after['identification_accuracy']:.4f} "
          f"EER {clean_after['eer']:.4f}")

    after = attack_curves(robust, X_imp, victims, threshold, epsilons, pgd_cfg, device)
    for name in ("fgsm", "pgd"):
        for r_b, r_a in zip(before["attacks"]["adversarial"][name], after[name]):
            print(f"[5.1 {name}] eps={r_b['epsilon']:.3f}: acceptance "
                  f"{r_b['acceptance_rate']:.4f} -> {r_a['acceptance_rate']:.4f}")

    # Epsilon curves before vs after
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for axi, name in zip(axes, ("fgsm", "pgd")):
        for rows, label, ls in [(before["attacks"]["adversarial"][name], "before (clean CNN)", "--"),
                                (after[name], "after (adversarial training)", "-")]:
            axi.plot([r["epsilon"] for r in rows], [r["acceptance_rate"] for r in rows],
                     marker="o", ls=ls, label=label)
        axi.axhline(clean_before["impostor_far_at_fixed_threshold"], color="gray", ls=":", lw=1)
        axi.set_xscale("log")
        axi.set_xlabel("epsilon (normalised units)")
        axi.set_title(name.upper())
        axi.legend(fontsize=8)
    axes[0].set_ylabel("impostor acceptance rate")
    axes[0].set_ylim(-0.02, 1.02)
    fig.suptitle("Targeted adversarial attacks before vs after adversarial training")
    fig_path = Path(cfg["paths"]["figures_dir"]) / "epsilon_before_after.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")

    # ---- 5.2 Challenge-response simulation ----
    task_set = build_task_clips(cfg, split)
    print(f"[5.2 challenge-response] task clips: {len(task_set['y'])} "
          f"(per class {np.bincount(task_set['y']).tolist()})")
    bands = {k: tuple(v) for k, v in cfg["features"]["bands"].items()}
    feats = log_band_power_features(
        task_set["X"], sfreq=cfg["dataset"]["expected_sfreq"],
        bands=bands, nperseg=cfg["features"]["welch_nperseg"])
    task_model, task_val_acc, X_val_task, y_val_task, val_idx = fit_task_classifier(
        feats, task_set["y"], seed=seeds["numpy"])
    print(f"[5.2 challenge-response] task classifier val accuracy: {task_val_acc:.4f}")

    # Identity score of each task clip under its owner's claimed class.
    # NOTE: these clips come from the TRAIN runs the CNN saw (eyes-open class is
    # literally training data; cue-locked clips overlap those runs). ID
    # acceptance is therefore optimistic — this sim isolates the task-challenge
    # mechanism, not the identity model.
    clean_cnn = _load_clean(cfg, device)
    probs_task = cnn_softmax_scores(clean_cnn, norm.transform(task_set["X"]), device=device)
    subjects_sorted = sorted(set(split["subject_ids_train"].tolist()))
    subj_to_cls = {int(s): i for i, s in enumerate(subjects_sorted)}
    cls_of_clip = np.array([subj_to_cls[int(s)] for s in task_set["subject_ids"]])
    true_scores = probs_task[np.arange(len(cls_of_clip)), cls_of_clip]
    id_scores_val = true_scores[val_idx]
    cr = challenge_response_simulation(task_model, X_val_task, y_val_task,
                                       id_scores_val, threshold, seed=seeds["numpy"])
    print(f"[5.2 challenge-response] replay success by library size: "
          f"{cr['replay_success_by_library_size']} (without challenge: 1.0)")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ks = sorted(cr["replay_success_by_library_size"], key=int)
    vals = [cr["replay_success_by_library_size"][k] for k in ks]
    ax.bar([f"k={k}" for k in ks], vals, color="#4C72B0")
    ax.axhline(1.0, color="gray", ls="--", label="no challenge-response (1.0)")
    ax.set_xlabel("attacker's task library size k (of 4 task types)")
    ax.set_ylabel("replay success rate")
    ax.set_title("Challenge-response vs replay (simulation)")
    ax.legend()
    fig_path = Path(cfg["paths"]["figures_dir"]) / "challenge_response.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")

    out = {
        "seeds": seeds,
        "threshold": threshold,
        "adversarial_training": {
            "train_seconds": round(train_s, 1),
            "epochs": len(history["train_loss"]),
            "pgd_eps": d_cfg["pgd_eps"],
            "clean_before": clean_before,
            "clean_after": clean_after,
            "attacks_before": before["attacks"]["adversarial"],
            "attacks_after": after,
        },
        "challenge_response": {
            "task_classifier_val_accuracy": task_val_acc,
            "n_task_clips": int(len(task_set["y"])),
            "task_clip_counts": np.bincount(task_set["y"]).tolist(),
            **cr,
        },
    }
    out_path = Path(cfg["paths"]["results_dir"]) / "defenses.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[run_defenses] results -> {out_path}")


def clean_eval_from_json(cnn_res: dict, threshold: float) -> dict:
    return {
        "identification_accuracy": cnn_res["identification_accuracy"],
        "eer": cnn_res["eer"],
        "auc": cnn_res["auc"],
        "impostor_far_at_fixed_threshold": cnn_res["far_at_eer"],
    }


def _load_clean(cfg: dict, device):
    ckpt = torch.load(Path(cfg["paths"]["models_dir"]) / "cnn.pt",
                      map_location=device, weights_only=False)
    model = EEGCNNSmall(**ckpt["meta"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    main()
