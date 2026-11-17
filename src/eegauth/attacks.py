"""Attacks against the authentication system.

Threat models implemented:
1. Impostor (zero-effort): submit your OWN brain data while claiming someone
   else's identity. Measured as FAR at the operating threshold (Phase 2/3).
2. Replay: resubmit previously recorded GENUINE clips of the victim. A static
   biometric cannot tell a live genuine user from a replayed recording of that
   user; acceptance rate equals the true-accept rate at the threshold.
3. Adversarial perturbation (targeted, white-box on the CNN): add a small
   L-inf-bounded noise to an impostor clip so that the system accepts it as
   the victim (FGSM one-step, PGD multi-step).

Adversarial attacks operate in the model's INPUT space, i.e. on per-channel
normalised clips (mean 0, std ~1 per electrode). Epsilons are therefore in
normalised units; the raw-volt equivalents are also reported.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["fgsm_linf", "pgd_linf", "targeted_attack_eval", "sample_attack_pairs", "DEFAULT_CLIP_RANGE"]

# Normalised clips are z-scored; +/-10 sigma is far outside plausible EEG.
DEFAULT_CLIP_RANGE = (-10.0, 10.0)


def fgsm_linf(
    model, x: torch.Tensor, target: torch.Tensor,
    eps: float, clip_range: Tuple[float, float] = DEFAULT_CLIP_RANGE,
) -> torch.Tensor:
    """Targeted Fast Gradient Sign Method (single step, L-inf ball).

    Targeted attacks MINIMISE the loss w.r.t. the input, so the step is
    x_adv = x - eps * sign(grad_x CE(model(x), target)).

    Returns adversarial inputs that move TOWARD the target class.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), target)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() - eps * grad.sign()
    return x_adv.clamp(*clip_range).detach()


def pgd_linf(
    model, x: torch.Tensor, target: torch.Tensor,
    eps: float, alpha: float, n_iter: int,
    clip_range: Tuple[float, float] = DEFAULT_CLIP_RANGE,
    random_start: bool = False,
) -> torch.Tensor:
    """Targeted Projected Gradient Descent (L-inf ball).

    Iterates: step by alpha in the sign-gradient direction, project back into
    the eps-ball around the clean input, and clip to the valid data range.
    """
    x_orig = x.clone().detach()
    if random_start:
        x_adv = x_orig + (2 * torch.rand_like(x_orig) - 1) * eps
    else:
        x_adv = x_orig.clone()
    for _ in range(n_iter):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), target)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_orig - eps), x_orig + eps)
            x_adv = x_adv.clamp(*clip_range)
    return x_adv.detach()


def sample_attack_pairs(
    X_test: np.ndarray, y_test: np.ndarray, n_pairs: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (impostor clip, victim class) pairs for targeted attacks.

    The impostor clip's true subject is always different from the victim.

    Returns:
        (X_imp, victims, idx): the impostor clips, their assigned victim class,
        and the indices of the chosen test clips.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y_test.max()) + 1
    idx = rng.choice(np.arange(len(y_test)), size=n_pairs, replace=False)
    X_imp = X_test[idx]
    victims = np.empty(n_pairs, dtype=np.int64)
    for k, i in enumerate(idx):
        true_cls = int(y_test[i])
        v = rng.integers(0, n_classes - 1)
        victims[k] = v + 1 if v >= true_cls else v
    assert (victims != y_test[idx]).all()
    return X_imp.astype(np.float32), victims, idx


def pgd_embed_cosine(
    model, x: torch.Tensor, template: torch.Tensor,
    eps: float, alpha: float, n_iter: int,
    clip_range: Tuple[float, float] = DEFAULT_CLIP_RANGE,
) -> torch.Tensor:
    """Targeted PGD in EMBEDDING space: maximise cosine(template, features(x)).

    Used against template/embedding-based verification (the login API):
    an impostor clip is perturbed until its embedding matches the victim's
    stored template.
    """
    x_orig = x.clone().detach()
    x_adv = x_orig.clone()
    for _ in range(n_iter):
        x_adv.requires_grad_(True)
        emb = model.features(x_adv)
        loss = -F.cosine_similarity(emb, template.expand(emb.shape[0], -1), dim=-1).sum()
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_orig - eps), x_orig + eps)
            x_adv = x_adv.clamp(*clip_range)
    return x_adv.detach()


def targeted_attack_eval(
    model,
    X_imp: np.ndarray,
    victims: np.ndarray,
    threshold: float,
    attack_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Run a targeted attack on all pairs and measure acceptance.

    A pair counts as accepted if the victim-class softmax score of the
    adversarial clip is >= threshold (the system's operating threshold).

    Returns acceptance rate, clean acceptance (eps=0 reference), and the mean
    perturbation size relative to the clean clip's signal (L2 and Linf).
    """
    n = X_imp.shape[0]
    accepted, clean_accepted = 0, 0
    l2_ratios, linf_ratios = [], []
    for start in range(0, n, batch_size):
        xb = torch.as_tensor(X_imp[start : start + batch_size], device=device)
        yb = torch.as_tensor(victims[start : start + batch_size], dtype=torch.long, device=device)
        x_adv = attack_fn(xb, yb)
        with torch.no_grad():
            scores = torch.softmax(model(xb), dim=1)[torch.arange(len(yb), device=device), yb]
            scores_adv = torch.softmax(model(x_adv), dim=1)[torch.arange(len(yb), device=device), yb]
        accepted += int((scores_adv >= threshold).sum().item())
        clean_accepted += int((scores >= threshold).sum().item())
        delta = (x_adv - xb).abs()
        l2_ratios.extend((delta.flatten(1).norm(dim=1) / xb.flatten(1).norm(dim=1)).tolist())
        linf_ratios.extend((delta.amax(dim=(1, 2)) / xb.abs().amax(dim=(1, 2))).tolist())
    return {
        "n_pairs": n,
        "acceptance_rate": accepted / n,
        "clean_acceptance_rate": clean_accepted / n,
        "mean_l2_perturbation_ratio": float(np.mean(l2_ratios)),
        "mean_linf_perturbation_ratio": float(np.mean(linf_ratios)),
    }
