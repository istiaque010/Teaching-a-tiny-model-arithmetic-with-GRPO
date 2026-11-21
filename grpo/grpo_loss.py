"""The GRPO policy objective (Phase 9). The exact variant used here, written out because implementations differ:

per response token t of response i (advantage A_i is shared by all tokens of the response):

    ratio_t   = exp(logp_new_t - logp_old_t)
    surr_t    = min(ratio_t * A_i,  clip(ratio_t, 1 - eps, 1 + eps) * A_i)
    kl_t      = exp(logp_ref_t - logp_new_t) - (logp_ref_t - logp_new_t) - 1        ("k3" estimator, >= 0)
    loss      = - sum_t (surr_t - beta * kl_t) / (number of response tokens in the minibatch)

* logp is taken on logits / temperature, the same temperature used to sample (so the sampler *is* the policy).
* Token-level averaging over the whole minibatch (not per-response averaging first): every token counts the same, so
  long responses are not down-weighted per token. This is the "token-level" variant (DAPO / TRL loss_type="bnpo").
* logp_old is recorded once at rollout time and kept fixed for all optimisation passes over that rollout batch.
"""
from __future__ import annotations

import torch


def grpo_loss(logp, old_logp, ref_logp, adv, mask, clip_eps: float = 0.2, beta: float = 0.0):
    """All token tensors (B, T); adv (B,); mask (B, T) bool (response tokens). ref_logp may be None when beta = 0.
    Returns (loss, stats dict)."""
    m = mask.float()
    n = m.sum().clamp(min=1.0)
    ratio = torch.exp(logp - old_logp)
    a = adv[:, None]
    s1 = ratio * a
    s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a
    surr = torch.minimum(s1, s2)
    obj = surr
    kl = torch.zeros_like(logp)
    if beta > 0 and ref_logp is not None:
        d = ref_logp - logp
        kl = torch.exp(d) - d - 1
        obj = surr - beta * kl
    loss = -(obj * m).sum() / n
    with torch.no_grad():
        clipped = ((s2 < s1) & (a != 0)).float()
        stats = {
            "loss": loss.item(),
            "kl_ref": (kl * m).sum().item() / n.item() if beta > 0 else float("nan"),
            "clip_frac": (clipped * m).sum().item() / n.item(),
            "approx_kl_old": ((old_logp - logp) * m).sum().item() / n.item(),
            "ratio_mean": (ratio * m).sum().item() / n.item(),
        }
    return loss, stats


def kl_k3(logp, ref_logp, mask) -> float:
    d = ref_logp - logp
    k = torch.exp(d) - d - 1
    m = mask.float()
    return ((k * m).sum() / m.sum().clamp(min=1)).item()
