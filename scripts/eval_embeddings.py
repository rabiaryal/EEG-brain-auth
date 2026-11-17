#!/usr/bin/env python
"""Evaluate embedding-cosine verification (used by the login API).

For each subject: template = mean embedding of TRAIN-run clips; test clips are
verified by cosine similarity against each subject template. Produces the
acceptance threshold used by the API.

Usage:
    python scripts/eval_embeddings.py [--checkpoint cnn]  # or cnn_robust

Writes results/embedding_verification.json with a "clean" and/or "robust" key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eegauth.config import load_config, set_seeds
from eegauth.data import build_dataset
from eegauth.metrics import roc_points, verification_metrics
from eegauth.models import EEGCNNSmall, get_device
from eegauth.train import PerChannelNormalizer


def embed(model, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(X[start : start + batch_size], device=device)
            out.append(model.features(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="cnn",
                       help="checkpoint stem in models/: 'cnn' (clean) or 'cnn_robust'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = set_seeds(cfg)
    device = get_device()

    ckpt_path = Path(cfg["paths"]["models_dir"]) / f"{args.checkpoint}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EEGCNNSmall(**ckpt["meta"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    split = build_dataset(cfg)
    norm = PerChannelNormalizer.load(Path(cfg["paths"]["models_dir"]) / "normalizer.npz")
    subjects = sorted(set(split["subject_ids_train"].tolist()))
    subj_to_cls = {int(s): i for i, s in enumerate(subjects)}

    # Templates from TRAIN-run clips only
    X_tr_e = embed(model, norm.transform(split["X_train"]), device)
    templates = {}
    for s in subjects:
        m = split["subject_ids_train"] == s
        templates[int(s)] = X_tr_e[m].mean(axis=0)

    # Test clips verified against every template
    X_te_e = embed(model, norm.transform(split["X_test"]), device)
    T = np.stack([templates[int(s)] for s in subjects])  # (n_subjects, d)
    T_norm = T / np.linalg.norm(T, axis=1, keepdims=True)
    E_norm = X_te_e / np.linalg.norm(X_te_e, axis=1, keepdims=True)
    sims = E_norm @ T_norm.T  # (n_test, n_subjects)

    cls_te = np.array([subj_to_cls[int(s)] for s in split["subject_ids_test"]])
    genuine = sims[np.arange(len(cls_te)), cls_te]
    mask = np.ones(sims.shape, dtype=bool)
    mask[np.arange(len(cls_te)), cls_te] = False
    impostor = sims[mask]
    m = verification_metrics(genuine, impostor)
    fpr, tpr, _ = roc_points(genuine, impostor)

    print(f"[eval_embeddings:{args.checkpoint}] EER {m.eer:.4f} | AUC {m.auc:.4f} "
          f"| threshold {m.eer_threshold:.6f}")
    out_path = Path(cfg["paths"]["results_dir"]) / "embedding_verification.json"
    data = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    data[args.checkpoint] = {
        **m.as_dict(),
        "n_genuine_scores": int(genuine.size),
        "n_impostor_scores": int(impostor.size),
        "seeds": seeds,
        "note": "template = mean TRAIN-run embedding; score = cosine similarity",
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[eval_embeddings] -> {out_path}")


if __name__ == "__main__":
    main()
