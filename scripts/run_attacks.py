#!/usr/bin/env python
"""Phase 4: evaluate the system under impostor, replay and adversarial attacks.

Usage:
    python scripts/run_attacks.py [--config configs/default.yaml]

Outputs:
    results/attacks.json
    results/figures/epsilon_curve.png        impostor acceptance vs epsilon
    results/figures/adversarial_example.png  clean vs PGD waveform (channel Fp1)
"""

from __future__ import annotations

import argparse
import json
import sys
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
from eegauth.models import EEGCNNSmall, get_device
from eegauth.train import PerChannelNormalizer, cnn_softmax_scores, genuine_impostor_scores


def load_trained_cnn(cfg: dict, device: torch.device):
    ckpt = torch.load(Path(cfg["paths"]["models_dir"]) / "cnn.pt", map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = EEGCNNSmall(**meta).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cnn-json", default=None, help="results json with the EER threshold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = set_seeds(cfg)
    device = get_device()
    cnn_json = Path(args.cnn_json) if args.cnn_json else Path(cfg["paths"]["results_dir"]) / "cnn.json"

    with open(cnn_json, encoding="utf-8") as f:
        cnn_res = json.load(f)
    threshold = float(cnn_res["eer_threshold"])
    print(f"[run_attacks] seeds {seeds} | device {device} | accept threshold (EER): {threshold:.6f}")

    split = build_dataset(cfg)
    norm = PerChannelNormalizer.load(Path(cfg["paths"]["models_dir"]) / "normalizer.npz")
    X_te = norm.transform(split["X_test"])
    y_te = split["y_test"]
    model, meta = load_trained_cnn(cfg, device)

    out: dict = {"seeds": seeds, "threshold": threshold, "attacks": {}}

    # ---- 4.1 Impostor (zero-effort) attack: FAR at the operating threshold ----
    probs = cnn_softmax_scores(model, X_te, device=device)
    genuine, impostor = genuine_impostor_scores(probs, y_te)
    far_clean = float(np.mean(impostor >= threshold))
    out["attacks"]["impostor_zero_effort"] = {
        "far_at_threshold": far_clean,
        "n_impostor_attempts": int(impostor.size),
        "note": "zero-effort impostor: wrong claimed identity, attacker's own EEG",
    }
    print(f"[4.1 impostor] FAR at threshold: {far_clean:.4f} over {impostor.size} attempts")

    # ---- 4.2 Replay attack: resubmit the victim's own recorded clips ----
    tpr_replay = float(np.mean(genuine >= threshold))
    # replay from ENROLMENT (train-run) clips of the victim (seen during training;
    # reported with that caveat)
    X_tr_n = norm.transform(split["X_train"])
    probs_tr = cnn_softmax_scores(model, X_tr_n, device=device)
    gen_tr, _ = genuine_impostor_scores(probs_tr, split["y_train"])
    tpr_replay_enroll = float(np.mean(gen_tr >= threshold))
    out["attacks"]["replay"] = {
        "acceptance_rate_test_recordings": tpr_replay,
        "acceptance_rate_enrollment_recordings": tpr_replay_enroll,
        "note": ("replay = resubmitting previously recorded genuine clips of the victim; "
                 "a static biometric cannot distinguish a live user from a replayed "
                 "recording of that user. Enrollment clips score higher because the "
                 "classifier was trained on them."),
    }
    print(f"[4.2 replay] acceptance: test-recordings {tpr_replay:.4f} | "
          f"enrollment-recordings {tpr_replay_enroll:.4f}")

    # ---- 4.3 Targeted adversarial attacks (FGSM, PGD) vs epsilon ----
    atk_cfg = cfg["attacks"]
    epsilons = list(atk_cfg["epsilons"])
    X_imp, victims, _ = sample_attack_pairs(
        X_te, y_te, n_pairs=int(atk_cfg["max_attack_pairs"]), seed=seeds["numpy"]
    )
    results_by_attack = {"fgsm": [], "pgd": []}
    for eps in epsilons:
        for name, fn in [
            ("fgsm", lambda xb, yb: fgsm_linf(model, xb, yb, eps=eps)),
            ("pgd", lambda xb, yb: pgd_linf(
                model, xb, yb, eps=eps, alpha=atk_cfg["pgd"]["alpha"],
                n_iter=atk_cfg["pgd"]["n_iter"])),
        ]:
            r = targeted_attack_eval(model, X_imp, victims, threshold, fn, device)
            r["epsilon"] = eps
            results_by_attack[name].append(r)
            print(f"[4.3 {name}] eps={eps:.3f} -> acceptance {r['acceptance_rate']:.4f} "
                  f"(L2 ratio {r['mean_l2_perturbation_ratio']:.4f})")
    out["attacks"]["adversarial"] = {
        "note": ("targeted white-box attacks on the CNN; epsilons are in per-channel-"
                 "normalised units (mean-channel std in raw volts is reported below)"),
        "raw_std_volts": float(np.mean(norm.std_)),
        "fgsm": results_by_attack["fgsm"],
        "pgd": results_by_attack["pgd"],
    }

    # Epsilon curve figure
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, ls in [("fgsm", "--"), ("pgd", "-")]:
        rows = results_by_attack[name]
        ax.plot([r["epsilon"] for r in rows], [r["acceptance_rate"] for r in rows],
                marker="o", ls=ls, label=f"{name.upper()} (targeted)")
    ax.axhline(far_clean, color="gray", ls=":", label=f"zero-effort FAR ({far_clean:.3f})")
    ax.set_xlabel("epsilon (normalised units; signal std ≈ 1)")
    ax.set_ylabel("impostor acceptance rate")
    ax.set_title("Targeted adversarial attacks vs epsilon (CNN)")
    ax.set_xscale("log")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    fig_path = Path(cfg["paths"]["figures_dir"]) / "epsilon_curve.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[run_attacks] figure -> {fig_path}")

    # Example adversarial clip (first pair, PGD at a mid epsilon)
    xb = torch.as_tensor(X_imp[:1], device=device)
    yb = torch.as_tensor(victims[:1], dtype=torch.long, device=device)
    eps_ex = epsilons[len(epsilons) // 2]
    x_adv = pgd_linf(model, xb, yb, eps=eps_ex, alpha=atk_cfg["pgd"]["alpha"],
                     n_iter=atk_cfg["pgd"]["n_iter"])[0, 0].cpu().numpy()
    clean = X_imp[0, 0]
    sfreq = cfg["dataset"]["expected_sfreq"]
    t = np.arange(clean.shape[-1]) / sfreq
    ch = meta and "Fp1"
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.plot(t, clean, label="clean impostor clip (ch. Fp1)")
    ax2.plot(t, x_adv, label=f"PGD eps={eps_ex}")
    ax2.plot(t, x_adv - clean, label="perturbation delta", lw=0.8)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("amplitude (normalised)")
    ax2.set_title(f"Targeted adversarial perturbation (claim as subject {int(victims[0]) + 1})")
    ax2.legend(fontsize=8)
    fig2_path = Path(cfg["paths"]["figures_dir"]) / "adversarial_example.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")

    out_path = Path(cfg["paths"]["results_dir"]) / "attacks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[run_attacks] results -> {out_path}")


if __name__ == "__main__":
    main()
