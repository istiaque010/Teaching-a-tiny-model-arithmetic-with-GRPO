"""GRPO training loop (Phase 10).

    python -m grpo.train_grpo --config grpo.yaml

One iteration:
  1. take `prompts_per_step` training questions;
  2. sample `group_size` responses to each from the current policy (temperature T, no top-k/top-p), recording the
     log-probability of every sampled token (= old-policy log-probs, kept fixed for this iteration);
  3. score every response with the verifiable reward; group-relative advantages;
  4. `epochs` passes over the rollout batch in minibatches of `minibatch` responses, each one clipped GRPO update
     (ratio against the *rollout-time* log-probs), optional KL to the frozen reference (= the starting policy).
Logged every iteration: reward and its components, zero-variance share, response length, entropy, KL to reference,
clip fraction, approximate KL old->new, gradient norm, rollout / optimiser seconds, memory. Validation accuracy every
`eval_every` iterations, with a checkpoint each time.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from . import CHECKPOINTS, METRICS, ckpt_path, load_config, save_json, write_jsonl
from . import dataset
from .advantages import group_advantages
from .evaluate import evaluate
from .grpo_loss import grpo_loss
from .policy import (decode_response, frozen_copy, load_policy, load_tokenizer, pack, sample, save_policy,
                     token_logprobs)
from .prompts import encode_prompt
from .rewards import reward

FIELDS = ["iter", "reward", "correct", "format", "zero_var", "resp_len", "entropy", "kl_ref", "clip_frac",
          "approx_kl", "grad_norm", "loss", "rollout_s", "opt_s", "train_s", "responses", "resp_tokens", "rss_mb",
          "val_greedy_acc", "val_sample_acc", "val_format"]


def rss_mb():
    import psutil

    info = psutil.Process(os.getpid()).memory_info()
    return getattr(info, "peak_wset", info.rss) / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = cfg["run"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(os.cpu_count())
    init = ckpt_path(cfg["init"])
    model, _ = load_policy(init)
    ref = frozen_copy(model) if cfg["beta"] > 0 else None
    tok = load_tokenizer()
    train = dataset.load("train")
    if cfg.get("difficulty", "all") != "all":
        train = [r for r in train if r["difficulty"] == cfg["difficulty"]]
    val = dataset.load("val")
    rng = random.Random(cfg["seed"])
    order = list(range(len(train)))
    rng.shuffle(order)
    gen = torch.Generator().manual_seed(cfg["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    out_dir = os.path.join(METRICS, "train", run)
    ck_dir = os.path.join(CHECKPOINTS, run)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    w = csv.DictWriter(log, FIELDS)
    w.writeheader()
    G, T = cfg["group_size"], cfg["temperature"]
    ptr, t_train, n_resp, n_tok = 0, 0.0, 0, 0
    samples_log, evals = [], []
    for it in range(cfg["iters"] + 1):
        if it % cfg["eval_every"] == 0 or it == cfg["iters"]:
            s, _, _ = evaluate(model, tok, val, n_samples=cfg["eval_samples"], seed=10_000 + it,
                               max_new=cfg["max_new"])
            evals.append({"iter": it, **{k: s[k] for k in ("greedy_acc", "sample_acc", "format_rate")}})
            w.writerow({"iter": it, "train_s": round(t_train, 1), "val_greedy_acc": s["greedy_acc"],
                        "val_sample_acc": s["sample_acc"], "val_format": s["format_rate"]})
            log.flush()
            print(f"[{run}] iter {it} val greedy {s['greedy_acc']:.3f} sample {s['sample_acc']:.3f} "
                  f"format {s['format_rate']:.3f}  train {t_train:.0f}s", flush=True)
            save_policy(model, os.path.join(ck_dir, f"iter_{it:04d}.pt"), run=run, iter=it, init=cfg["init"])
            if it == cfg["iters"]:
                break
        # ---- rollout
        t0 = time.perf_counter()
        if ptr + cfg["prompts_per_step"] > len(order):
            rng.shuffle(order)
            ptr = 0
        rows = [train[i] for i in order[ptr:ptr + cfg["prompts_per_step"]]]
        ptr += cfg["prompts_per_step"]
        prompts = [encode_prompt(tok, r["question"]) for r in rows for _ in range(G)]
        model.eval()
        resp, old_lp = sample(model, prompts, cfg["max_new"], tok.eos_id, tok.pad_id, temperature=T, generator=gen)
        texts = [decode_response(tok, r) for r in resp]
        rw = [reward(t, rows[i // G]["answer"], cfg["w_correct"], cfg["w_format"]) for i, t in enumerate(texts)]
        R = torch.tensor([x["reward"] for x in rw])
        adv, zero_var = group_advantages(R, G, scale=cfg.get("scale_std", True))
        ids, attn, rmask = pack(prompts, resp, tok.pad_id)
        old = torch.zeros(ids.shape[0], ids.shape[1] - 1)
        P = max(len(p) for p in prompts)
        for i, lp in enumerate(old_lp):
            old[i, P - 1:P - 1 + len(lp)] = torch.tensor(lp)
        ref_lp = None
        if ref is not None:
            with torch.no_grad():
                ref_lp, _, _ = token_logprobs(ref, ids, attn, rmask, temperature=T)
        t_roll = time.perf_counter() - t0
        # ---- optimise
        t1 = time.perf_counter()
        model.train()
        N = len(prompts)
        stats, gnorms, ents = [], [], []
        active = [i for i in range(N) if adv[i] != 0] if cfg.get("skip_zero_adv", True) else list(range(N))
        for _ in range(cfg["epochs"]):
            perm = active[:]
            rng.shuffle(perm)
            for b in range(0, len(perm), cfg["minibatch"]):
                idx = torch.tensor(perm[b:b + cfg["minibatch"]])
                lp, ent, m = token_logprobs(model, ids[idx], attn[idx], rmask[idx], temperature=T,
                                            with_entropy=True)
                loss, st = grpo_loss(lp, old[idx], None if ref_lp is None else ref_lp[idx], adv[idx], m,
                                     cfg["clip_eps"], cfg["beta"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item())
                for gp in opt.param_groups:
                    gp["lr"] = cfg["lr"] * min(1.0, (it + 1) / max(1, cfg["warmup"]))
                opt.step()
                stats.append(st)
                ents.append(((ent * m).sum() / m.sum().clamp(min=1)).item())
        t_opt = time.perf_counter() - t1
        t_train += t_roll + t_opt
        n_resp += N
        n_tok += sum(len(r) for r in resp)
        mean = lambda k: float(np.mean([s[k] for s in stats])) if stats else float("nan")  # noqa: E731
        row = {"iter": it, "reward": R.mean().item(), "correct": np.mean([x["correctness"] for x in rw]),
               "format": np.mean([x["format"] for x in rw]), "zero_var": zero_var,
               "resp_len": np.mean([len(r) for r in resp]), "entropy": float(np.mean(ents)) if ents else "",
               "kl_ref": mean("kl_ref") if cfg["beta"] > 0 else "", "clip_frac": mean("clip_frac"),
               "approx_kl": mean("approx_kl_old"), "grad_norm": float(np.mean(gnorms)) if gnorms else "",
               "loss": mean("loss"), "rollout_s": round(t_roll, 2), "opt_s": round(t_opt, 2),
               "train_s": round(t_train, 1), "responses": n_resp, "resp_tokens": n_tok, "rss_mb": round(rss_mb())}
        w.writerow(row)
        log.flush()
        if it % 10 == 0:
            print(f"[{run}] iter {it} reward {row['reward']:.3f} correct {row['correct']:.3f} "
                  f"zero_var {zero_var:.2f} len {row['resp_len']:.1f} clip {row['clip_frac']:.3f} "
                  f"kl {row['kl_ref']} roll {t_roll:.1f}s opt {t_opt:.1f}s", flush=True)
        if it % 25 == 0:
            for j in range(0, min(N, 2 * G)):
                samples_log.append({"iter": it, "question": rows[j // G]["question"], "answer": rows[j // G]["answer"],
                                    "response": texts[j], "reward": rw[j]["reward"], "adv": adv[j].item()})
    save_policy(model, os.path.join(ck_dir, "final.pt"), run=run, iter=cfg["iters"], init=cfg["init"])
    write_jsonl(samples_log, out_dir, "rollout_samples.jsonl")
    save_json({"run": run, "config": cfg, "train_seconds": t_train, "responses": n_resp, "response_tokens": n_tok,
               "evals": evals, "peak_rss_mb": rss_mb()}, out_dir, "summary.json")
    log.close()


if __name__ == "__main__":
    main()
