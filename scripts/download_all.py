#!/usr/bin/env python
"""Download EEGMMIDB runs for subjects 1..109 and screen each file's header.

Writes results/full109/screening.json listing, per subject, whether all needed
runs have 160 Hz / >=64 EEG channels. Subjects failing the screen are excluded
from the 109-subject config (and the reason is logged in AGENT.md).

Usage: python scripts/download_all.py [--runs 1 2 3 4 5 6 7]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegauth.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full109.yaml")
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"]) / "eegbci"
    report: dict = {}
    for s in range(1, 110):
        try:
            paths = mne.datasets.eegbci.load_data(s, args.runs, path=str(data_dir),
                                                  update_path=False, verbose="ERROR")
            info = {}
            for p in paths:
                raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
                info[Path(p).stem] = {"sfreq": raw.info["sfreq"], "n_ch": len(raw.ch_names),
                                      "dur_s": round(raw.n_times / raw.info["sfreq"], 1)}
            bad = [k for k, v in info.items() if abs(v["sfreq"] - 160.0) > 1e-6 or v["n_ch"] < 64]
            report[str(s)] = {"ok": not bad, "bad_runs": bad, "runs": info}
            print(f"S{s:03d} {'OK' if not bad else 'BAD ' + str(bad)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            report[str(s)] = {"ok": False, "error": repr(exc)}
            print(f"S{s:03d} FAILED {exc!r}", flush=True)
    out = Path(cfg["paths"]["results_dir"]) / "screening.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    bad_subjects = sorted(int(k) for k, v in report.items() if not v["ok"])
    print("excluded:", bad_subjects)


if __name__ == "__main__":
    main()
