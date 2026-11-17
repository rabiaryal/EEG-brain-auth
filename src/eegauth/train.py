"""Model training and evaluation helpers (Phase 2 baseline; Phase 3 extends).

Identification task: predict which subject a clip belongs to (closed-set).
Verification task: score-based accept/reject of a *claim*, evaluated with the
zero-effort impostor protocol in eegauth.metrics.

Anti-leakage rules enforced here:
- Scalers / normalisation are fitted on TRAIN features only.
- Train and test clips always come from disjoint runs (guaranteed upstream by
  eegauth.data.split_by_runs and tested there).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

__all__ = [
    "fit_baseline", "identification_accuracy", "score_matrix",
    "genuine_impostor_scores", "PerChannelNormalizer", "train_cnn",
]


def fit_baseline(
    X_train: np.ndarray, y_train: np.ndarray, kind: str = "logreg", **kwargs: Any
) -> Pipeline:
    """Fit a baseline classifier: StandardScaler -> LogReg or RBF-SVM.

    The scaler is fitted on training data only (inside the pipeline).

    Args:
        X_train: (n_train, n_features) feature matrix.
        y_train: (n_train,) subject-class indices.
        kind: "logreg" or "svm".
        **kwargs: passed to the underlying estimator (e.g. C).
    """
    if kind == "logreg":
        est = LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=2000,
            solver="lbfgs",
            random_state=kwargs.get("random_state", 42),
        )
    elif kind == "svm":
        est = SVC(
            C=kwargs.get("C", 10.0),
            gamma=kwargs.get("gamma", "scale"),
            kernel="rbf",
            decision_function_shape="ovr",
            cache_size=1000,
        )
    else:
        raise ValueError(f"unknown baseline kind: {kind}")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    pipe.fit(X_train, y_train)
    return pipe


def identification_accuracy(
    model: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Top-1 closed-set identification accuracy on the test clips."""
    return float(np.mean(model.predict(X_test) == y_test))


def score_matrix(
    model: Pipeline, X: np.ndarray, n_classes: int
) -> np.ndarray:
    """Per-class scores for every clip, shape (n_clips, n_classes).

    Uses predict_proba when available, else decision_function. For the
    verification protocol, the genuine score of clip i is scores[i, y_i] and
    its impostor scores are scores[i, c] for c != y_i.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        scores = model.decision_function(X)
    assert scores.shape[1] == n_classes
    return scores


def genuine_impostor_scores(
    scores: np.ndarray, y_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Split per-class scores into genuine and zero-effort impostor scores.

    Args:
        scores: (n_clips, n_classes) per-class scores.
        y_true: (n_clips,) true subject-class index per clip.

    Returns:
        (genuine, impostor): genuine has one score per clip; impostor has
        (n_clips, n_classes - 1) scores (every wrong claim per clip).
    """
    n = scores.shape[0]
    idx = np.arange(n)
    genuine = scores[idx, y_true]
    mask = np.ones(scores.shape, dtype=bool)
    mask[idx, y_true] = False
    impostor = scores[mask].reshape(n, -1)
    return genuine, impostor.ravel()


# --------------------------------------------------------------------------- #
# Per-channel normalisation (fitted on TRAIN clips ONLY)
# --------------------------------------------------------------------------- #
class PerChannelNormalizer:
    """Per-electrode z-scoring, fitted on training clips only.

    For each channel c: x'_c = (x_c - mean_c) / max(std_c, eps), where mean_c
    and std_c are computed over ALL training clips and time samples of that
    channel. Test/val clips are transformed with the SAME statistics.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PerChannelNormalizer":
        """X: (n_clips, n_channels, n_samples) training clips."""
        if X.ndim != 3:
            raise ValueError(f"expected (n, C, T), got {X.shape}")
        self.mean_ = X.mean(axis=(0, 2))
        std = X.std(axis=(0, 2))
        self.std_ = np.maximum(std, self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("PerChannelNormalizer must be fit() before transform()")
        return ((X - self.mean_[None, :, None]) / self.std_[None, :, None]).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, std=self.std_, eps=self.eps)

    @classmethod
    def load(cls, path: str | Path) -> "PerChannelNormalizer":
        with np.load(path) as z:
            norm = cls(eps=float(z["eps"]))
            norm.mean_ = z["mean"]
            norm.std_ = z["std"]
        return norm


# --------------------------------------------------------------------------- #
# CNN training with early stopping (validation from TRAIN runs only)
# --------------------------------------------------------------------------- #
def train_cnn(
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
    verbose: bool = True,
) -> Tuple[Dict[str, list], Any]:
    """Train the CNN with cross-entropy + Adam and early stopping.

    The validation split is taken from the TRAIN clips only (train runs never
    contain test runs), so no test data touches training.

    Returns:
        (history, best_state_dict) where history has per-epoch train/val loss
        and val accuracy; best_state_dict is the lowest-val-loss checkpoint.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(val_fraction * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    if verbose:
        print(f"[train_cnn] train={len(tr_idx)} clips, val={len(val_idx)} clips "
              f"(both from TRAIN runs only), device={device}")

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
            optimizer.zero_grad()
            loss = criterion(model(xb), y_tr_t[b])
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
            print(f"[train_cnn] epoch {epoch:3d} | train {train_loss:.4f} "
                  f"| val {val_loss:.4f} | val_acc {val_acc:.4f}")
        if epochs_no_improve >= patience:
            if verbose:
                print(f"[train_cnn] early stop at epoch {epoch} "
                      f"(best val loss {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_state


def cnn_softmax_scores(
    model, X: np.ndarray, batch_size: int = 256, device=None
) -> np.ndarray:
    """Softmax probability matrix (n_clips, n_classes) from a trained CNN."""
    import torch

    model.eval()
    device = device or next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(X[start : start + batch_size], device=device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)
