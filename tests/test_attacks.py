"""Unit tests for FGSM/PGD attacks (tiny model, CPU)."""

import numpy as np
import pytest
import torch

from eegauth.attacks import fgsm_linf, pgd_linf, sample_attack_pairs, targeted_attack_eval
from eegauth.models import EEGCNNSmall


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = EEGCNNSmall(n_channels=8, n_samples=64, n_classes=3, n_filters=8)
    model.eval()
    x = torch.randn(6, 8, 64)
    target = torch.tensor([2, 2, 2, 0, 0, 0])
    return model, x, target


def test_fgsm_epsilon_zero_is_identity(tiny):
    model, x, target = tiny
    x_adv = fgsm_linf(model, x, target, eps=0.0)
    assert torch.allclose(x_adv, x, atol=1e-7)


def test_pgd_epsilon_zero_is_identity(tiny):
    model, x, target = tiny
    x_adv = pgd_linf(model, x, target, eps=0.0, alpha=0.01, n_iter=3)
    assert torch.allclose(x_adv, x, atol=1e-7)


def test_pgd_respects_epsilon_ball(tiny):
    model, x, target = tiny
    eps = 0.05
    x_adv = pgd_linf(model, x, target, eps=eps, alpha=0.02, n_iter=5)
    delta = (x_adv - x).abs().max().item()
    assert delta <= eps + 1e-6, f"delta {delta} exceeds eps {eps}"
    assert x_adv.shape == x.shape


def test_pgd_improves_target_confidence(tiny):
    model, x, target = tiny
    with torch.no_grad():
        before = torch.softmax(model(x), dim=1)[torch.arange(6), target]
    x_adv = pgd_linf(model, x, target, eps=0.5, alpha=0.1, n_iter=20)
    with torch.no_grad():
        after = torch.softmax(model(x_adv), dim=1)[torch.arange(6), target]
    assert (after >= before).float().mean() >= 0.5, "targeted PGD did not move scores toward target"


def test_fgsm_single_step_moves_scores(tiny):
    model, x, target = tiny
    x_adv = fgsm_linf(model, x, target, eps=0.5)
    with torch.no_grad():
        before = torch.softmax(model(x), dim=1)[torch.arange(6), target]
        after = torch.softmax(model(x_adv), dim=1)[torch.arange(6), target]
    assert (after >= before).float().mean() >= 0.5


def test_sample_attack_pairs_no_self_pairs():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    X = np.random.default_rng(0).standard_normal((10, 4, 32)).astype(np.float32)
    X_imp, victims, idx = sample_attack_pairs(X, y, n_pairs=5, seed=1)
    assert X_imp.shape == (5, 4, 32)
    assert (victims != y[idx]).all(), "victim must differ from the clip's true subject"
    assert len(np.unique(idx)) == 5


def test_attack_eval_metrics(tiny):
    model, x, target = tiny
    device = torch.device("cpu")
    X = x.numpy()
    y = target.numpy()
    res = targeted_attack_eval(
        model, X, y, threshold=0.5,
        attack_fn=lambda xb, yb: fgsm_linf(model, xb, yb, eps=0.5),
        device=device, batch_size=4,
    )
    assert res["n_pairs"] == 6
    assert 0.0 <= res["acceptance_rate"] <= 1.0
    assert res["mean_l2_perturbation_ratio"] >= 0.0


def test_pgd_embed_cosine_respects_ball_and_raises_similarity(tiny):
    import torch.nn.functional as F
    from eegauth.attacks import pgd_embed_cosine

    model, x, _ = tiny
    with torch.no_grad():
        template = model.features(x[:1])[0]
        template = template / template.norm()
    eps = 0.3
    x_adv = pgd_embed_cosine(model, x, template[None, :], eps=eps, alpha=0.1, n_iter=10)
    assert (x_adv - x).abs().max().item() <= eps + 1e-6
    with torch.no_grad():
        before = F.cosine_similarity(model.features(x), template[None, :].expand(6, -1), dim=-1)
        after = F.cosine_similarity(model.features(x_adv), template[None, :].expand(6, -1), dim=-1)
    # sample 0 produced the template (cosine already 1.0), so only check the others
    assert (after[1:] > before[1:]).all(), "embedding PGD must raise similarity to the target"
