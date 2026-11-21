"""Group-relative advantages (Phase 8)."""
import torch

import grpo  # noqa: F401
from grpo.advantages import group_advantages


def test_guide_example():
    adv, zero = group_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0]), 4)
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=1e-4)
    assert zero == 0.0


def test_equal_rewards_give_zero_and_are_counted():
    r = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.1, 0.1, 0.1])
    adv, zero = group_advantages(r, 4)
    assert torch.all(adv[:8] == 0)
    assert torch.isfinite(adv).all()
    assert abs(zero - 2 / 3) < 1e-6


def test_zero_mean_unit_std_per_group():
    torch.manual_seed(0)
    r = torch.randint(0, 2, (64,)).float() + 0.1 * torch.randint(0, 2, (64,)).float()
    adv, _ = group_advantages(r, 8)
    a = adv.view(-1, 8)
    for g, row in zip(r.view(-1, 8), a):
        if g.std() > 0:
            assert abs(row.mean().item()) < 1e-5
            assert abs(row.std(unbiased=False).item() - 1) < 1e-3


def test_order_inside_group():
    adv, _ = group_advantages(torch.tensor([0.0, 1.1, 0.1, 1.0]), 4)
    assert adv[1] > adv[3] > adv[2] > adv[0]


def test_unscaled_option():
    adv, _ = group_advantages(torch.tensor([1.0, 0.0]), 2, scale=False)
    assert torch.allclose(adv, torch.tensor([0.5, -0.5]))
