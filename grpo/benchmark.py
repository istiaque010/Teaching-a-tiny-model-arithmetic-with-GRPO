"""Inference latency / throughput and training-compute summary (Phase 13).

    python -m grpo.benchmark --systems base sft grpo ...

CPU only (no GPU on this PC), fp32, all cores. For each system:
  * latency: batch 1, greedy, the first 100 test questions, after 10 warm-up answers; median and p95 over 3 repeats
    (seconds per answer, and per generated token);
  * throughput: batch 64 sampled answers (T = 1.0), 3 repeats, answers per second.
All systems share the architecture, so differences come only from how many tokens they write.
A background-load check (other processes' CPU use) is recorded with every measurement.
Training compute (wall-clock, responses, tokens, peak memory) is read from each run's summary.json.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import psutil
import torch

from . import METRICS, ckpt_path, load_json, save_json
from . import dataset
from .policy import load_policy, load_tokenizer, sample
from .prompts import encode_prompt


def ckpt_of(name):
    return ckpt_path(name)


def other_cpu():
    me = psutil.Process(os.getpid())
    tot = psutil.cpu_percent(interval=0.5)
    mine = me.cpu_percent(interval=0.5) / psutil.cpu_count()
    return max(0.0, tot - mine)


def bench(name, rows, tok, repeats=3, ckpt=None):
    model, _ = load_policy(ckpt_of(ckpt or name))
    enc = [encode_prompt(tok, r["question"]) for r in rows]
    for p in enc[:10]:
        sample(model, [p], 64, tok.eos_id, tok.pad_id, greedy=True)
    lat, per_tok, ntok = [], [], []
    for _ in range(repeats):
        for p in enc:
            t0 = time.perf_counter()
            r, _ = sample(model, [p], 64, tok.eos_id, tok.pad_id, greedy=True)
            dt = time.perf_counter() - t0
            lat.append(dt)
            ntok.append(len(r[0]))
            per_tok.append(dt / max(1, len(r[0])))
    thr = []
    g = torch.Generator().manual_seed(0)
    batch = [enc[i % len(enc)] for i in range(64)]
    for _ in range(repeats):
        t0 = time.perf_counter()
        sample(model, batch, 64, tok.eos_id, tok.pad_id, generator=g)
        thr.append(64 / (time.perf_counter() - t0))
    return {"name": name, "latency_median_s": statistics.median(lat), "latency_p95_s": float(np.percentile(lat, 95)),
            "ms_per_token": 1000 * statistics.median(per_tok), "median_tokens": statistics.median(ntok),
            "batch64_answers_per_s": statistics.median(thr), "batch64_runs": thr,
            "background_cpu_pct": other_cpu()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    tok = load_tokenizer()
    rows = dataset.load("test")[:100]
    out = {}
    for spec in args.systems:  # "name" or "name=checkpoint" (e.g. sft_1000=sft/step_01000)
        n, _, ck = spec.partition("=")
        out[n] = bench(n, rows, tok, ckpt=ck or None)
        print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[n].items() if k != "batch64_runs"})
    train = {}
    for r in args.runs:
        s = load_json(METRICS, "train", r, "summary.json")
        train[r] = {k: s.get(k) for k in ("train_seconds", "responses", "response_tokens", "examples",
                                          "peak_rss_mb")}
    save_json({"hardware": {"cpu_count": os.cpu_count(), "device": "cpu", "precision": "fp32",
                            "torch_threads": torch.get_num_threads()},
               "inference": out, "training": train}, METRICS, "benchmark.json")


if __name__ == "__main__":
    main()
