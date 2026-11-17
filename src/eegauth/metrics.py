"""Biometric verification metrics: ROC, EER, FAR and FRR.

Protocol (zero-effort impostor, score-based):
- For each test clip we obtain a *genuine score* (score assigned to the clip's
  true claimed identity) and *impostor scores* (scores assigned to every other
  claimed identity).
- Acceptance rule: accept claim if score >= threshold.
- FAR(threshold) = fraction of impostor scores >= threshold.
- FRR(threshold) = fraction of genuine scores <  threshold.
- EER: the operating point where FAR = FRR, obtained by linear interpolation
  on the ROC curve (FPR vs TPR) at the crossing with the anti-diagonal
  (FPR + TPR = 1, i.e. FPR = FRR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

__all__ = ["roc_points", "eer_from_roc", "VerificationMetrics", "verification_metrics"]


def roc_points(
    genuine: np.ndarray, impostor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FPR/TPR/thresholds for genuine vs impostor scores (positives = genuine).

    Uses sklearn's roc_curve; thresholds are the score values (descending).
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("need at least one genuine and one impostor score")
    y = np.concatenate([np.ones(genuine.size), np.zeros(impostor.size)])
    s = np.concatenate([genuine, impostor])
    fpr, tpr, thresholds = roc_curve(y, s)
    return fpr, tpr, thresholds


def eer_from_roc(
    fpr: np.ndarray, tpr: np.ndarray
) -> Tuple[float, float]:
    """EER (and the FAR at EER) by interpolating the ROC curve.

    The EER is the point where FPR = FRR = 1 - TPR, i.e. FPR + TPR = 1.
    We scan the ROC polyline (which is monotone) for the segment crossing the
    anti-diagonal and linearly interpolate inside that segment.

    Returns:
        (eer, threshold_far) where threshold_far equals the FAR (= FRR) value
        at the crossing point.
    """
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    total = fpr + tpr - 1.0  # = 0 on the anti-diagonal

    # handle perfect separation at the start of the curve
    if total[0] >= 0:
        return float(max(fpr[0], 1e-12)), float(fpr[0])

    idx = np.where(total >= 0)[0]
    if idx.size == 0:  # curve never reaches the diagonal: degenerate system
        return 1.0, 1.0
    i = int(idx[0])
    j = i - 1
    df, dt = fpr[i] - fpr[j], tpr[i] - tpr[j]
    denom = df + dt
    if abs(denom) < 1e-15:
        u = 0.0
    else:
        u = (1.0 - (fpr[j] + tpr[j])) / denom
    u = float(np.clip(u, 0.0, 1.0))
    eer = fpr[j] + u * df
    return float(eer), float(eer)


@dataclass
class VerificationMetrics:
    """Result of a verification evaluation."""

    eer: float
    far_at_eer: float
    frr_at_eer: float
    eer_threshold: float
    auc: float

    def as_dict(self) -> dict:
        return {
            "eer": self.eer,
            "far_at_eer": self.far_at_eer,
            "frr_at_eer": self.frr_at_eer,
            "eer_threshold": self.eer_threshold,
            "auc": self.auc,
        }


def verification_metrics(
    genuine: np.ndarray, impostor: np.ndarray
) -> VerificationMetrics:
    """Compute ROC, AUC, EER, and FAR/FRR at the EER threshold.

    Args:
        genuine: scores for correct claims (higher = more genuine).
        impostor: scores for wrong claims (zero-effort impostors).
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    fpr, tpr, thresholds = roc_points(genuine, impostor)
    eer, _ = eer_from_roc(fpr, tpr)

    # Find the ROC point closest to the EER point (eer, 1-eer) in (FPR, TPR)
    # space -- needed because many thresholds can share the same FPR.
    target_tpr = 1.0 - eer
    k = int(np.argmin(np.hypot(fpr - eer, tpr - target_tpr)))
    eer_threshold = float(thresholds[k])

    # FAR/FRR exactly at that threshold with the accept rule score >= t.
    far = float(np.mean(impostor >= eer_threshold))
    frr = float(np.mean(genuine < eer_threshold))
    auc = float(roc_auc_score(
        np.concatenate([np.ones(genuine.size), np.zeros(impostor.size)]),
        np.concatenate([genuine, impostor]),
    ))
    return VerificationMetrics(
        eer=eer, far_at_eer=far, frr_at_eer=frr,
        eer_threshold=eer_threshold, auc=auc,
    )
