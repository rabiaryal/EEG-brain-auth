#!/usr/bin/env python
"""Build the train/test dataset per configs/default.yaml and log statistics.

Usage:
    python scripts/build_dataset.py [--config configs/default.yaml] [--no-cache]

Writes results/data_stats.json and prints a stats table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eegauth.config import load_config, set_seeds
from eegauth.data import build_dataset, dataset_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-cache", action="store_true", help="ignore existing split cache")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = set_seeds(cfg)
    print(f"[build_dataset] seeds: {seeds}")

    split = build_dataset(cfg, use_cache=not args.no_cache)
    stats = dataset_stats(split)
    stats["seeds"] = seeds
    stats["filter"] = cfg["filter"]
    stats["epochs"] = {"length_s": cfg["epochs"]["length_s"], "overlap_s": cfg["epochs"]["overlap_s"]}
    stats["train_runs_config"] = cfg["dataset"]["train_runs"]
    stats["test_runs_config"] = cfg["dataset"]["test_runs"]

    # Sanity assertions: the split must be leak-free before we log it.
    tr, te = set(stats["train_runs"]), set(stats["test_runs"])
    assert tr.isdisjoint(te), f"leakage: {tr & te}"

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "data_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[build_dataset] stats written to {out}")
    print(f"  subjects:          {stats['n_subjects']}")
    print(f"  n_channels x samples: {stats['n_channels']} x {stats['n_samples_per_clip']}")
    print(f"  train clips:       {stats['n_train_clips']}  (runs {stats['train_runs']})")
    print(f"  test clips:        {stats['n_test_clips']}  (runs {stats['test_runs']})")
    print(f"  clips/subject train: min {stats['train_clips_min']}, max {stats['train_clips_max']}")
    print(f"  clips/subject test:  min {stats['test_clips_min']}, max {stats['test_clips_max']}")


if __name__ == "__main__":
    main()
