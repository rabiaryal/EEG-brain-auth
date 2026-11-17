"""Defenses: adversarial training and (simulated) challenge-response.

1. Adversarial training: train the CNN on a mix of clean and on-the-fly PGD
   examples (targeted at the clip's own class? No — targeted at a RANDOM wrong
   class is closer to the attack threat, but standard adversarial training
   minimises loss on adversarially perturbed versions of each example,
   i.e. untargeted PGD around the clean input). We follow the standard recipe:
   x_adv = PGD(x, y, eps, alpha, n_iter) maximising CE(model(x_adv), y),
   then train on the mix. Re-running the Phase 4.3 attacks measures robustness.
2. Challenge-response (stretch, SIMULATION): the system requests a random
   task (eyes-open rest / motor tasks from the dataset's own runs) and must
   verify BOTH the task and the identity. A replay attacker with only a
   partial library of the victim's recordings must then have a clip of the
   requested task. Clearly labelled as a simulation using dataset tasks.
"""

from __future__ import annotations

import copy
import itertools
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "train_cnn_adversarial",
    "task_labels_from_runs",
    "build_task_clips",
    "fit_task_classifier",
    "challenge_response_simulation",
]

# legacy name kept for imports in tests
RUN_TO_TASK = {1: "eyes_open", 3: "task1", 4: "task2", 5: "task3"}


def train_cnn_adversarial(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    val_fraction: float,
    device,
    seed: int = 42,
    weight_decay: float = 1e-4,
    adv_fraction: float = 0.5,
    pgd_alpha: float = 0.01,
    pgd_n_iter: int = 10,
    pgd_eps: float = 0.05,
    verbose: bool = True,
) -> Tuple[Dict[str, list], Dict]:
    """Adversarial training: each batch mixes clean clips with PGD examples.

    PGD examples are generated ON THE FLY with the current model weights
    (untargeted, maximising CE at the clip's own label), eps=pgd_eps.
    Early stopping is on the CLEAN validation loss so the clean-accuracy
    cost of the defense is measured fairly.

    Returns (history, best_state_dict).
    """
    import torch
    import torch.nn as nn

    from eegauth.attacks import pgd_linf

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(val_fraction * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    if verbose:
        print(f"[adv_train] train={len(tr_idx)}, val={len(val_idx)} (train-run clips only)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, epochs_no_improve = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(X_tr))
        running = 0.0
        for start in range(0, len(order), batch_size):
            b = order[start : start + batch_size]
            xb = torch.as_tensor(X_tr[b], device=device)
            yb = y_tr_t[b]
            n_adv = int(round(adv_fraction * len(b)))
            if n_adv > 0:
                # untargeted PGD: maximise CE at the clip's own label
                x_adv = pgd_linf(
                    model, xb[:n_adv], yb[:n_adv], eps=pgd_eps, alpha=pgd_alpha,
                    n_iter=pgd_n_iter,
                )
                xb = torch.cat([x_adv, xb[n_adv:]], dim=0)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(b)
        train_loss = running / len(order)

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.as_tensor(X_val, device=device))
            val_loss = criterion(val_logits, y_val_t).item()
            val_acc = float((val_logits.argmax(1) == y_val_t).float().mean().item())
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val - 1e-5:
            best_val, epochs_no_improve = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"[adv_train] epoch {epoch:3d} | train {train_loss:.4f} "
                  f"| val {val_loss:.4f} | val_acc {val_acc:.4f}")
        if epochs_no_improve >= patience:
            if verbose:
                print(f"[adv_train] early stop at epoch {epoch} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_state


# --------------------------------------------------------------------------- #
# Challenge-response (simulation)
# --------------------------------------------------------------------------- #
# Task classes: 0 = eyes-open rest (uniform clips of train run 1), 1-3 = the
# cue-locked movement tasks of train runs 3/4/5.
TASK_NAMES = ["eyes_open", "task1", "task2", "task3"]
MOTOR_RUN_TO_TASK = {3: 1, 4: 2, 5: 3}


def build_task_clips(cfg: dict, split: dict) -> dict:
    """Build the challenge-response task clip set (from TRAIN runs only).

    eyes-open rest: uniform 2 s clips of run 1 (the whole run IS rest).
    tasks 1-3: cue-locked clips around T1/T2 annotation onsets of runs 3/4/5.

    Returns dict with X (n, C, T), y (n,), subject_ids (n,), run_ids (n,).
    """
    import numpy as np

    from eegauth.data import load_run_task_clips

    flt, ep = cfg["filter"], cfg["epochs"]
    mask = split["run_ids_train"] == 1
    X_parts = [split["X_train"][mask]]
    y_parts = [np.zeros(int(mask.sum()), dtype=np.int64)]
    sid_parts = [split["subject_ids_train"][mask]]
    rid_parts = [np.full(int(mask.sum()), 1, dtype=np.int64)]
    for run, task in MOTOR_RUN_TO_TASK.items():
        per_subject = []
        for subject in sorted(set(split["subject_ids_train"].tolist())):
            X = load_run_task_clips(
                int(subject), run,
                l_freq=flt["l_freq"], h_freq=flt["h_freq"], length_s=ep["length_s"],
                cache_dir=cfg["paths"]["cache_dir"], data_dir=cfg["paths"].get("data_dir"),
            )
            per_subject.append(X)
        X = np.concatenate(per_subject, axis=0)
        X_parts.append(X)
        y_parts.append(np.full(len(X), task, dtype=np.int64))
        sid_parts.append(np.repeat(np.array(sorted(set(split["subject_ids_train"].tolist())),
                                        dtype=np.int64), [len(p) for p in per_subject]))
        rid_parts.append(np.full(len(X), run, dtype=np.int64))
    return {
        "X": np.concatenate(X_parts, axis=0),
        "y": np.concatenate(y_parts, axis=0),
        "subject_ids": np.concatenate(sid_parts, axis=0),
        "run_ids": np.concatenate(rid_parts, axis=0),
    }


def task_labels_from_runs(run_ids: np.ndarray) -> np.ndarray:
    """Map run numbers to the 4 task classes present in the train runs."""
    mapping = {1: 0, **MOTOR_RUN_TO_TASK}
    labels = np.array([mapping.get(int(r), -1) for r in run_ids])
    if (labels < 0).any():
        raise ValueError(f"runs outside the task set: {sorted(set(run_ids.tolist()))}")
    return labels


def fit_task_classifier(
    features: np.ndarray, labels: np.ndarray, seed: int = 42, val_fraction: float = 0.2
) -> Tuple:
    """Fit a LogReg task classifier; returns (model, val_acc, X_val, y_val, val_idx)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    n_val = max(1, int(val_fraction * len(labels)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=2000))])
    pipe.fit(features[tr_idx], labels[tr_idx])
    val_acc = float(accuracy_score(labels[val_idx], pipe.predict(features[val_idx])))
    return pipe, val_acc, features[val_idx], labels[val_idx], val_idx


def challenge_response_simulation(
    task_model,
    X_val_task: np.ndarray,
    y_val_task: np.ndarray,
    id_scores_for_clips: np.ndarray,
    threshold: float,
    n_draws: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """Simulate replay attacks with and without a challenge-response step.

    Setup (SIMULATION with dataset tasks — the victim is not physically present):
    - The system requests a task T uniformly from the 4 dataset task classes.
    - The replay attacker owns a library: clips of the victim from a random
      subset S of the task classes, |S| = k (the attacker previously recorded
      only k kinds of task).
    - Login succeeds iff (a) T in S (the attacker has material for the requested
      task), (b) the task classifier accepts the submitted clip as T, and
    (c) the identity score >= threshold.
    Without challenge-response, any victim clip is accepted (replay = genuine
    biometric signal), so success = 1.0 for every k.

    Args:
        task_model: fitted task classifier (predict on features).
        X_val_task: task features of the held-out clips (the "library").
        y_val_task: true task labels for those clips.
        id_scores_for_clips: identity score of each library clip under its
            owner's claimed identity (CNN softmax at true class).
        threshold: identity acceptance threshold.

    Returns:
        dict: replay success rate per library size k, plus component stats.
    """
    rng = np.random.default_rng(seed)
    n_tasks = len(TASK_NAMES)
    clips_by_task: Dict[int, List[int]] = {t: [] for t in range(n_tasks)}
    for i, t in enumerate(y_val_task):
        clips_by_task[int(t)].append(i)

    # P(task classifier accepts the true task) and P(id accept | true task)
    task_ok = np.array(
        [task_model.predict(X_val_task[i : i + 1])[0] == int(t)
         for i, t in enumerate(y_val_task)]
    )
    p_task_accept = float(task_ok.mean())
    id_ok = id_scores_for_clips >= threshold
    p_id_accept = float(id_ok.mean())

    success_by_k: Dict[int, float] = {}
    for k in range(1, n_tasks + 1):
        subsets = list(itertools.combinations(range(n_tasks), k))
        draws_per_subset = max(1, n_draws // len(subsets))
        successes = 0
        trials = 0
        for subset in subsets:
            for _ in range(draws_per_subset):
                t_req = int(rng.integers(0, n_tasks))
                trials += 1
                if t_req not in subset or not clips_by_task[t_req]:
                    continue
                # submit a random library clip of the requested task
                i = int(rng.choice(clips_by_task[t_req]))
                if task_model.predict(X_val_task[i : i + 1])[0] == t_req and id_ok[i]:
                    successes += 1
        success_by_k[k] = successes / trials if trials else 0.0
    return {
        "p_task_classifier_accept": p_task_accept,
        "p_identity_accept": p_id_accept,
        "replay_success_by_library_size": {str(k): v for k, v in success_by_k.items()},
        "replay_success_without_challenge": 1.0,
        "note": ("simulation: victim not physically present; requestable tasks are the "
                 "dataset's own train-run tasks {eyes_open, task1, task2, task3}; "
                 "library = held-out clips of each victim per task"),
    }
