#!/usr/bin/env python
"""End-to-end demo of the login API (in-process, via FastAPI TestClient).

Steps (all on real EEGMMIDB clips, enrollment = TRAIN runs, probes = TEST runs):
  1. enroll subject A
  2. verify A with unseen genuine clips        -> expected ACCEPT (+ JWT)
  3. verify B claiming to be A (zero-effort)    -> expected REJECT
  4. white-box adversarial clips (B perturbed toward A's stored template with
     PGD on the embedding cosine) submitted to the API running the CLEAN model
     and the ADVERSARIALLY-TRAINED model (each attacked adaptively).

Writes results/demo_api.json. Every number printed comes from this run.

Usage: python scripts/demo_api.py [--n-attack 30] [--eps 0.1]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from eegauth.api import create_app
from eegauth.attacks import pgd_embed_cosine
from eegauth.config import load_config, set_seeds
from eegauth.data import build_dataset


def npy(arr: np.ndarray) -> dict:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return {"clips_file": ("clips.npy", buf.getvalue(), "application/octet-stream")}


def craft_adversarial_raw(app, victim_template: np.ndarray, clips_raw: np.ndarray,
                          eps: float, alpha: float, n_iter: int) -> np.ndarray:
    """PGD on the deployed model: perturb RAW clips so their embedding matches the template."""
    norm, model, dev = app.state.normalizer, app.state.model, app.state.device
    x = torch.as_tensor(norm.transform(clips_raw), device=dev)
    t = torch.as_tensor(victim_template / np.linalg.norm(victim_template), device=dev).float()
    x_adv = pgd_embed_cosine(model, x, t[None, :], eps=eps, alpha=alpha, n_iter=n_iter)
    delta_norm = (x_adv - x).cpu().numpy()
    return (clips_raw + delta_norm * norm.std_[None, :, None]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-attack", type=int, default=30)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--n-iter", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = set_seeds(cfg)
    split = build_dataset(cfg)
    sid_tr, sid_te = split["subject_ids_train"], split["subject_ids_test"]
    subj = sorted(set(sid_tr.tolist()))
    s_a, s_b = int(subj[0]), int(subj[1])
    print(f"[demo] A = subject {s_a}, B = subject {s_b}; seeds {seeds}")

    enroll_clips = split["X_train"][sid_tr == s_a]
    genuine_probe = split["X_test"][sid_te == s_a][:20]
    impostor_probe = split["X_test"][sid_te == s_b][: args.n_attack]

    report: dict = {"subject_A": s_a, "subject_B": s_b, "seeds": seeds}
    with tempfile.TemporaryDirectory() as td:
        clean = create_app(args.config, "cnn", Path(td) / "clean")
        robust = create_app(args.config, "cnn_robust", Path(td) / "robust")
        c_clean, c_robust = TestClient(clean), TestClient(robust)

        # 1. enroll
        r = c_clean.post("/enroll", data={"user_id": "A"}, files=npy(enroll_clips))
        print(f"[1] enroll A ({len(enroll_clips)} clips): {r.status_code} {r.json()}")
        assert r.status_code == 200
        report["enroll"] = r.json()

        # 2. genuine
        r = c_clean.post("/verify", data={"user_id": "A"}, files=npy(genuine_probe))
        b = r.json()
        print(f"[2] A verifies as A: accepted={b['accepted']} score={b['score']:.4f} "
              f"(threshold {b['threshold']:.4f}) token={'yes' if 'token' in b else 'no'}")
        report["genuine_verify"] = {k: v for k, v in b.items() if k != "token"}
        report["genuine_verify"]["token_issued"] = "token" in b

        # 3. zero-effort impostor
        r = c_clean.post("/verify", data={"user_id": "A"}, files=npy(impostor_probe[:20]))
        b = r.json()
        print(f"[3] B claims A: accepted={b['accepted']} score={b['score']:.4f} "
              f"token={'yes' if 'token' in b else 'no'}")
        report["impostor_verify"] = {k: v for k, v in b.items() if k != "token"}
        report["impostor_verify"]["token_issued"] = "token" in b

        # Per-clip zero-effort acceptance for the attack set (single-clip requests)
        def per_clip_accept(client: TestClient, clips: np.ndarray) -> float:
            acc = 0
            for clip in clips:
                rr = client.post("/verify", data={"user_id": "A"}, files=npy(clip[None]))
                acc += int(rr.json()["accepted"])
            return acc / len(clips)

        # 4. adversarial, before vs after defense (adaptive white-box on each model)
        report["adversarial"] = {"eps_normalised": args.eps, "n_clips": len(impostor_probe),
                                 "pgd": {"alpha": args.alpha, "n_iter": args.n_iter}}
        for name, client, app in (("clean_model", c_clean, clean), ("robust_model", c_robust, robust)):
            if name == "robust_model":
                r = client.post("/enroll", data={"user_id": "A"}, files=npy(enroll_clips))
                assert r.status_code == 200
            template = np.load(Path(td) / name.split("_")[0] / "A" / "template.npy")
            zero_effort = per_clip_accept(client, impostor_probe)
            adv = craft_adversarial_raw(app, template, impostor_probe, args.eps, args.alpha, args.n_iter)
            adv_rate = per_clip_accept(client, adv)
            genuine_rate = per_clip_accept(client, split["X_test"][sid_te == s_a][: args.n_attack])
            raw_ratio = float(np.mean(np.linalg.norm((adv - impostor_probe).reshape(len(adv), -1), axis=1)
                                       / np.linalg.norm(impostor_probe.reshape(len(adv), -1), axis=1)))
            print(f"[4] {name}: genuine accept {genuine_rate:.2%} | zero-effort B→A accept {zero_effort:.2%} "
                  f"| adversarial B→A accept {adv_rate:.2%} (L2 perturbation/signal {raw_ratio:.3f})")
            report["adversarial"][name] = {
                "genuine_accept_rate": genuine_rate,
                "zero_effort_accept_rate": zero_effort,
                "adversarial_accept_rate": adv_rate,
                "mean_l2_perturbation_ratio_raw": raw_ratio,
                "threshold": app.state.threshold,
            }

    out = Path(cfg["paths"]["results_dir"]) / "demo_api.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[demo] saved {out}")


if __name__ == "__main__":
    main()
