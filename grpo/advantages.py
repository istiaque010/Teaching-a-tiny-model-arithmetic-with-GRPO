"""Group-relative advantages (Phase 8).

For the G responses to one question with rewards r_1..r_G:

    A_i = (r_i - mean(r)) / (std(r) + eps)        std = population standard deviation (divide by G)

A response better than its siblings gets A > 0 (pushed up), a worse one A < 0. If every reward in the group is the
same (all right or all wrong), std = 0 and every A_i is exactly 0: that question carries no learning signal.
The share of such zero-variance groups is reported at every update.
"""
from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6, scale: bool = True):
    """rewards: (N,) with N = n_groups * group_size, groups contiguous. Returns (advantages (N,), zero_var_frac)."""
    r = rewards.float().view(-1, group_size)
    mean = r.mean(1, keepdim=True)
    std = r.std(1, unbiased=False, keepdim=True)
    zero = std.squeeze(1) < 1e-8
    adv = r - mean
    if scale:
        adv = adv / (std + eps)
    adv[zero] = 0.0
    return adv.view(-1), zero.float().mean().item()
