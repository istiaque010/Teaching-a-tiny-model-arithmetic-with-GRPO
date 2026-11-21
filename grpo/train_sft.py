"""Supervised fine-tuning warm-up (Phase 5) and the matched "more SFT" control (Phase 11).

    python -m grpo.train_sft --config sft.yaml                        # base -> checkpoints/sft
    python -m grpo.train_sft --config sft_extra.yaml                  # sft  -> checkpoints/sft_extra

Cross-entropy on the *response* tokens only (worked steps + final-answer line + <EOS>), never on the question.
Only the train split is used; validation is evaluated every `eval_every` updates (greedy + sampled accuracy) and a
checkpoint is kept at each evaluation. Logs: results/metrics/train/<run>/log.csv and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from . import CHECKPOINTS, METRICS, ckpt_path, load_config, save_json
from . import dataset
from .evaluate import evaluate
from .policy import forward_padded, load_policy, load_tokenizer, pack, save_policy
from .prompts import encode_prompt, encode_target


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    if cfg.get("schedule", "cosine") == "constant":
        return cfg["lr"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * p))


def sft_loss(model, tok, batch_rows):
    ps = [encode_prompt(tok, r["question"]) for r in batch_rows]
    ts = [encode_target(tok, r["solution"], r["answer"]) for r in batch_rows]
    ids, attn, resp = pack(ps, ts, tok.pad_id)
    logits = forward_padded(model, ids, attn)[:, :-1]
    m = resp[:, :-1]
    tgt = ids[:, 1:]
    loss = F.cross_entropy(logits[m], tgt[m])
    return loss, int(m.sum())


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
    model.train()  # dropout is already 0; train() only matters for consistency
    tok = load_tokenizer()
    train = dataset.load("train")
    val = dataset.load("val")
    rng = random.Random(cfg["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    out_dir = os.path.join(METRICS, "train", run)
    ck_dir = os.path.join(CHECKPOINTS, run)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "lr", "loss", "tokens", "secs", "val_greedy_acc", "val_sample_acc", "val_format"])
    order = list(range(len(train)))
    rng.shuffle(order)
    ptr, tokens, t_train, best = 0, 0, 0.0, None
    evals = []
    for step in range(cfg["steps"] + 1):
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            model.eval()
            s, _, _ = evaluate(model, tok, val, n_samples=cfg["eval_samples"], seed=step)
            model.train()
            evals.append({"step": step, **{k: s[k] for k in ("greedy_acc", "sample_acc", "format_rate")}})
            w.writerow([step, "", "", tokens, round(t_train, 1), s["greedy_acc"], s["sample_acc"], s["format_rate"]])
            log.flush()
            print(f"[{run}] step {step} val greedy {s['greedy_acc']:.3f} sample {s['sample_acc']:.3f} "
                  f"format {s['format_rate']:.3f}  train {t_train:.0f}s", flush=True)
            save_policy(model, os.path.join(ck_dir, f"step_{step:05d}.pt"), run=run, step=step, init=cfg["init"])
            if step == cfg["steps"]:
                break
        t0 = time.perf_counter()
        if ptr + cfg["batch"] > len(order):
            rng.shuffle(order)
            ptr = 0
        rows = [train[i] for i in order[ptr:ptr + cfg["batch"]]]
        ptr += cfg["batch"]
        lr = lr_at(step, cfg)
        for gp in opt.param_groups:
            gp["lr"] = lr
        loss, n = sft_loss(model, tok, rows)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += n
        t_train += time.perf_counter() - t0
        if step % 50 == 0:
            w.writerow([step, lr, round(loss.item(), 4), tokens, round(t_train, 1), "", "", ""])
            log.flush()
            print(f"[{run}] step {step} loss {loss.item():.4f} lr {lr:.2e}", flush=True)
    save_policy(model, os.path.join(ck_dir, "final.pt"), run=run, step=cfg["steps"], init=cfg["init"])
    save_json({"run": run, "config": cfg, "train_seconds": t_train, "response_tokens": tokens,
               "examples": cfg["steps"] * cfg["batch"], "evals": evals}, out_dir, "summary.json")
    log.close()


if __name__ == "__main__":
    main()
