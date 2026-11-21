"""GRPO objective (Phase 9): ratio, clipping, KL, masking, gradient direction."""
import torch

import grpo  # noqa: F401
from grpo.grpo_loss import grpo_loss, kl_k3


def _toy(B=4, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    logp = torch.randn(B, T, generator=g).clamp(-5, 0) - 0.1
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 4:] = False
    return logp, mask


def test_on_policy_loss_is_minus_mean_advantage():
    logp, mask = _toy()
    adv = torch.tensor([1.0, -1.0, 0.5, 0.0])
    loss, st = grpo_loss(logp, logp.clone(), None, adv, mask)
    want = -(adv[:, None] * mask).sum() / mask.sum()
    assert torch.allclose(loss, want)
    assert st["clip_frac"] == 0 and abs(st["ratio_mean"] - 1) < 1e-6


def test_gradient_pushes_good_up_bad_down():
    logp, mask = _toy()
    p = logp.clone().requires_grad_(True)
    adv = torch.tensor([1.0, -1.0, 0.0, 0.0])
    loss, _ = grpo_loss(p, logp.detach(), None, adv, mask)
    loss.backward()
    assert (p.grad[0][mask[0]] < 0).all()  # descent raises log p of the good response
    assert (p.grad[1] > 0).all()           # and lowers the bad one
    assert (p.grad[2:] == 0).all()


def test_masked_tokens_have_no_effect():
    logp, mask = _toy()
    p = logp.clone().requires_grad_(True)
    loss, _ = grpo_loss(p, logp.detach(), None, torch.ones(4), mask)
    loss.backward()
    assert (p.grad[0, 4:] == 0).all()


def test_clipping_stops_gradient_beyond_eps():
    old = torch.zeros(1, 3)
    new = torch.log(torch.tensor([[1.5, 1.1, 0.5]])).requires_grad_(True)  # ratios 1.5, 1.1, 0.5
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss, st = grpo_loss(new, old, None, torch.tensor([1.0]), mask, clip_eps=0.2)
    loss.backward()
    assert new.grad[0, 0] == 0          # A>0 and ratio > 1.2: clipped, no further push
    assert new.grad[0, 1] != 0
    assert new.grad[0, 2] != 0          # A>0 and ratio < 0.8: min keeps the unclipped term
    assert abs(st["clip_frac"] - 1 / 3) < 1e-6


def test_kl_penalty_zero_at_reference_and_positive_elsewhere():
    logp, mask = _toy()
    assert abs(kl_k3(logp, logp, mask)) < 1e-7
    assert kl_k3(logp, logp - 0.3, mask) > 0
    adv = torch.zeros(4)
    l0, _ = grpo_loss(logp, logp, logp - 0.3, adv, mask, beta=0.0)
    l1, st = grpo_loss(logp, logp, logp - 0.3, adv, mask, beta=0.1)
    assert l0.item() == 0 and l1.item() > 0 and st["kl_ref"] > 0


def test_finite_with_extreme_values():
    logp = torch.full((2, 3), -30.0)
    old = torch.zeros(2, 3)
    loss, _ = grpo_loss(logp, old, old, torch.tensor([1.0, -1.0]), torch.ones(2, 3, dtype=torch.bool), beta=0.04)
    assert torch.isfinite(loss)
