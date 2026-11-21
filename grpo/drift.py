"""How far each trained policy moved from a reference policy (Phase 14, H9), measured the same way for every run.

    python -m grpo.drift --ref sft/step_01000 --systems grpo_early_s0 early_beta0 ...

The training log has KL to the reference only for runs with beta > 0 (the beta = 0 run never loads a reference), so
for a fair comparison drift is measured after training: KL(policy || reference) per response token, estimated on the
policy's own samples (T = 1.0) for the first 250 test questions x 8 samples:

    KL = E_{y ~ policy} [ log policy(y_t) - log ref(y_t) ]   averaged over response tokens

This is the exact KL's unbiased Monte-Carlo estimate (no k3 approximation needed because samples come from the policy).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import METRICS, ckpt_path, save_json
from . import dataset
from .policy import load_policy, load_tokenizer, pack, sample, token_logprobs
from .prompts import encode_prompt


@torch.no_grad()
def drift(name, ref, rows, tok, n_samples=8, batch=32, seed=0):
    model, _ = load_policy(ckpt_path(name))
    g = torch.Generator().manual_seed(seed)
    per_resp = []
    prompts = [encode_prompt(tok, r["question"]) for r in rows for _ in range(n_samples)]
    for i in range(0, len(prompts), batch):
        p = prompts[i:i + batch]
        resp, _ = sample(model, p, 64, tok.eos_id, tok.pad_id, generator=g)
        ids, attn, rmask = pack(p, resp, tok.pad_id)
        lp, _, m = token_logprobs(model, ids, attn, rmask)
        lr, _, _ = token_logprobs(ref, ids, attn, rmask)
        m = m.float()
        d = ((lp - lr) * m).sum(1) / m.sum(1).clamp(min=1)
        per_resp += d.tolist()
    per_resp = np.array(per_resp)
    return {"kl_per_token_mean": float(per_resp.mean()), "kl_per_token_se": float(per_resp.std() / np.sqrt(len(per_resp))),
            "n_responses": len(per_resp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="sft/step_01000")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--out", default="drift_early.json")
    args = ap.parse_args()
    tok = load_tokenizer()
    ref, _ = load_policy(ckpt_path(args.ref))
    rows = dataset.load("test")[:args.limit]
    out = {"ref": args.ref, "systems": {}}
    for n in args.systems:
        out["systems"][n] = drift(n, ref, rows, tok)
        print(f"{n:>16}: KL to {args.ref} = {out['systems'][n]['kl_per_token_mean']:.4f} "
              f"± {out['systems'][n]['kl_per_token_se']:.4f} per token")
    save_json(out, METRICS, args.out)


if __name__ == "__main__":
    main()
