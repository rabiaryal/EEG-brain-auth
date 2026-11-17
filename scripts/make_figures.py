#!/usr/bin/env python
"""Phase 7.2: final figures and results table from saved JSON metrics only.

Reads every results directory that exists (results/ = 20 subjects,
results/full109/ = all screened subjects) and writes, to results/final/:
    final_results.md              markdown results table (all numbers from JSON)
    fig_epsilon_curve.png         targeted-attack acceptance vs epsilon (+ robust model)
    fig_before_after.png          adversarial-training before/after bars
    fig_models.png                identification accuracy / EER per model
    fig_replay_challenge.png      replay success with / without challenge-response

No number is typed by hand: everything is read from the JSON files produced by
the run_* scripts.

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "final"
SETS = {"20 subjects": ROOT / "results", "all screened subjects": ROOT / "results" / "full109"}
COLORS = {"20 subjects": "#2a6fb0", "all screened subjects": "#d9822b"}


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial proportion k/n (k may be a rate*n)."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def collect() -> dict:
    sets = {}
    for label, d in SETS.items():
        if not (d / "cnn.json").exists():
            continue
        stats = load(d, "data_stats.json") or {}
        sets[label] = {
            "dir": d,
            "stats": stats,
            "baseline": load(d, "baseline.json"),
            "cnn": load(d, "cnn.json"),
            "attacks": load(d, "attacks.json"),
            "defenses": load(d, "defenses.json"),
            "embed": load(d, "embedding_verification.json"),
        }
    return sets


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}%"


def adv_at(rows: list[dict], eps: float) -> dict:
    return min(rows, key=lambda r: abs(r["epsilon"] - eps))


def fig_epsilon(sets: dict) -> None:
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4), squeeze=False)
    for ax, (label, S) in zip(axes[0], sets.items()):
        adv = S["attacks"]["attacks"]["adversarial"]
        for name, ls, mk in (("fgsm", "--", "s"), ("pgd", "-", "o")):
            rows = adv[name]
            n = rows[0]["n_pairs"]
            eps = [r["epsilon"] for r in rows]
            rate = np.array([r["acceptance_rate"] for r in rows])
            ci = np.array([wilson(r * n, n) for r in rate])
            ax.plot(eps, rate, ls, marker=mk, color="#2a6fb0", label=f"{name.upper()} (clean CNN)")
            if name == "pgd":
                ax.fill_between(eps, ci[:, 0], ci[:, 1], color="#2a6fb0", alpha=0.15)
        d = (S["defenses"] or {}).get("adversarial_training")
        if d:
            rows = d["attacks_after"]["pgd"]
            ax.plot([r["epsilon"] for r in rows], [r["acceptance_rate"] for r in rows],
                    "-", marker="^", color="#d9822b", label="PGD (adv.-trained CNN)")
        ax.axhline(S["attacks"]["attacks"]["impostor_zero_effort"]["far_at_threshold"],
                   color="gray", ls=":", label="zero-effort FAR")
        ax.set_xscale("log")
        ax.set_xlabel("epsilon (L-inf, per-channel z-score units)")
        ax.set_ylabel("impostor acceptance rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Targeted attack, {label} (n={n} pairs)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_epsilon_curve.png", dpi=150)
    plt.close(fig)


def fig_before_after(sets: dict, eps_list=(0.02, 0.05, 0.1)) -> None:
    items = [(l, S) for l, S in sets.items() if (S["defenses"] or {}).get("adversarial_training")]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(5.2 * len(items), 4), squeeze=False)
    for ax, (label, S) in zip(axes[0], items):
        d = S["defenses"]["adversarial_training"]
        x = np.arange(len(eps_list))
        w = 0.38
        for j, (key, name, col) in enumerate((("attacks_before", "before (standard training)", "#2a6fb0"),
                                              ("attacks_after", "after (adversarial training)", "#d9822b"))):
            vals = [adv_at(d[key]["pgd"], e)["acceptance_rate"] for e in eps_list]
            ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=name)
            for xi, v in zip(x + (j - 0.5) * w, vals):
                ax.text(xi, v + 0.01, f"{100 * v:.0f}%", ha="center", fontsize=8)
        ax.set_xticks(x, [f"eps={e}" for e in eps_list])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("PGD impostor acceptance")
        ca, cb = d["clean_after"], d["clean_before"]
        ax.set_title(f"{label}\nclean EER {pct(cb['eer'])} -> {pct(ca['eer'])}", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_before_after.png", dpi=150)
    plt.close(fig)


def fig_models(sets: dict) -> None:
    names = ["LogReg (band power)", "RBF-SVM (band power)", "CNN (softmax)", "CNN (embedding cosine)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.38
    for j, (label, S) in enumerate(sets.items()):
        acc, eer = [], []
        b = S["baseline"]["models"]
        acc += [b["logreg"]["identification_accuracy"], b["svm"]["identification_accuracy"],
                S["cnn"]["identification_accuracy"], np.nan]
        eer += [b["logreg"]["eer"], b["svm"]["eer"], S["cnn"]["eer"],
                (S["embed"] or {}).get("cnn", {}).get("eer", np.nan)]
        x = np.arange(len(names)) + (j - 0.5) * w
        axes[0].bar(x, acc, w, color=COLORS[label], label=label)
        axes[1].bar(x, eer, w, color=COLORS[label], label=label)
    for ax, t in zip(axes, ("Identification accuracy (test runs)", "Verification EER (lower is better)")):
        ax.set_xticks(np.arange(len(names)), names, rotation=25, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_models.png", dpi=150)
    plt.close(fig)


def fig_replay(sets: dict) -> None:
    items = [(l, S) for l, S in sets.items() if (S["defenses"] or {}).get("challenge_response")]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, S in items:
        cr = S["defenses"]["challenge_response"]
        ks = sorted(int(k) for k in cr["replay_success_by_library_size"])
        ys = [cr["replay_success_by_library_size"][str(k)] for k in ks]
        ax.plot(ks, ys, "-o", color=COLORS[label], label=f"{label} (task acc {pct(cr['task_classifier_val_accuracy'], 1)})")
    ax.axhline(1.0, color="gray", ls=":", label="replay without challenge")
    ax.set_xlabel("task types the attacker has recorded of the victim (k of 4)")
    ax.set_ylabel("replay success")
    ax.set_title("Challenge-response vs replay (simulation on dataset tasks)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_replay_challenge.png", dpi=150)
    plt.close(fig)


def results_table(sets: dict) -> str:
    labels = list(sets)
    rows: list[tuple[str, list[str]]] = []

    def add(name: str, fn) -> None:
        vals = []
        for l in labels:
            try:
                vals.append(fn(sets[l]))
            except (KeyError, TypeError, IndexError):
                vals.append("n/a")
        rows.append((name, vals))

    def subj(S):
        return str(S["stats"].get("n_subjects", "?"))

    add("Subjects", subj)
    add("Train / test clips", lambda S: f"{S['stats']['n_train_clips']} / {S['stats']['n_test_clips']}")
    add("LogReg identification acc", lambda S: pct(S["baseline"]["models"]["logreg"]["identification_accuracy"]))
    add("LogReg EER (AUC)", lambda S: f"{pct(S['baseline']['models']['logreg']['eer'])} ({S['baseline']['models']['logreg']['auc']:.4f})")
    add("SVM identification acc", lambda S: pct(S["baseline"]["models"]["svm"]["identification_accuracy"]))
    add("SVM EER (AUC)", lambda S: f"{pct(S['baseline']['models']['svm']['eer'])} ({S['baseline']['models']['svm']['auc']:.4f})")
    add("CNN identification acc", lambda S: pct(S["cnn"]["identification_accuracy"]))
    add("CNN EER (AUC), softmax score", lambda S: f"{pct(S['cnn']['eer'])} ({S['cnn']['auc']:.4f})")
    add("CNN EER, embedding-cosine (API verifier)", lambda S: pct(S["embed"]["cnn"]["eer"]))
    add("Zero-effort impostor FAR @ EER thr.", lambda S: pct(S["attacks"]["attacks"]["impostor_zero_effort"]["far_at_threshold"]))
    add("Replay acceptance (unseen test recordings)", lambda S: pct(S["attacks"]["attacks"]["replay"]["acceptance_rate_test_recordings"]))
    add("Replay acceptance (enrollment recordings)", lambda S: pct(S["attacks"]["attacks"]["replay"]["acceptance_rate_enrollment_recordings"]))
    for e in (0.02, 0.05, 0.1):
        add(f"FGSM eps={e} acceptance", lambda S, e=e: pct(adv_at(S["attacks"]["attacks"]["adversarial"]["fgsm"], e)["acceptance_rate"], 1))
        add(f"PGD eps={e} acceptance", lambda S, e=e: pct(adv_at(S["attacks"]["attacks"]["adversarial"]["pgd"], e)["acceptance_rate"], 1))
    add("PGD eps=0.1 L2 perturbation / signal", lambda S: f"{adv_at(S['attacks']['attacks']['adversarial']['pgd'], 0.1)['mean_l2_perturbation_ratio']:.3f}")
    add("Adv. training: clean id. acc (before -> after)", lambda S: f"{pct(S['defenses']['adversarial_training']['clean_before']['identification_accuracy'])} -> {pct(S['defenses']['adversarial_training']['clean_after']['identification_accuracy'])}")
    add("Adv. training: clean EER (before -> after)", lambda S: f"{pct(S['defenses']['adversarial_training']['clean_before']['eer'])} -> {pct(S['defenses']['adversarial_training']['clean_after']['eer'])}")
    for e in (0.02, 0.05, 0.1):
        add(f"PGD eps={e}: before -> after adv. training", lambda S, e=e: f"{pct(adv_at(S['defenses']['adversarial_training']['attacks_before']['pgd'], e)['acceptance_rate'], 1)} -> {pct(adv_at(S['defenses']['adversarial_training']['attacks_after']['pgd'], e)['acceptance_rate'], 1)}")
    add("Task-classifier accuracy (challenge-response)", lambda S: pct(S["defenses"]["challenge_response"]["task_classifier_val_accuracy"], 1))
    for k in (1, 2, 3, 4):
        add(f"Replay success with challenge, k={k}", lambda S, k=k: pct(S["defenses"]["challenge_response"]["replay_success_by_library_size"][str(k)], 1))
    add("Attack pairs per epsilon", lambda S: str(S["attacks"]["attacks"]["adversarial"]["pgd"][0]["n_pairs"]))

    head = "| Metric | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n"
    return head + "\n".join(f"| {n} | " + " | ".join(v) + " |" for n, v in rows) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = collect()
    if not sets:
        raise SystemExit("no results found; run the run_* scripts first")
    fig_epsilon(sets)
    fig_before_after(sets)
    fig_models(sets)
    fig_replay(sets)
    table = results_table(sets)
    (OUT / "final_results.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"[make_figures] wrote figures + table to {OUT}")


if __name__ == "__main__":
    main()
