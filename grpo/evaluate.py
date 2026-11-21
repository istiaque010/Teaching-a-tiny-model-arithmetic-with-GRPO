"""Generate answers and grade them with the same code for every system (Phases 4, 11, 12).

    python -m grpo.evaluate --ckpt <path|base> --name <system> --split test --samples 16 --greedy

For each question: one greedy answer (optional) and k sampled answers at temperature T (default 1.0, no top-k /
top-p, i.e. the policy's own distribution). Records are written to results/generations/<split>/<name>.jsonl and a
summary to results/metrics/eval/<split>/<name>.json:

  greedy_acc      accuracy of the greedy answer
  sample_acc      mean accuracy over the k sampled answers (= expected pass@1 when sampling)
  pass@k          unbiased estimator (Chen et al., 2021) from n = k samples, for k in 1, 2, 4, 8, 16
  format_rate     share of sampled answers with a parseable final answer
  per-family / per-difficulty accuracies, median response tokens, latency per answer
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time

import numpy as np
import torch

from . import GENERATIONS, ckpt_path, METRICS, save_json, write_jsonl
from . import dataset
from .answer_parser import grade
from .policy import decode_response, load_policy, load_tokenizer, sample
from .prompts import encode_prompt

MAX_NEW = 64


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def generate_answers(model, tok, rows, n_samples: int = 1, temperature: float = 1.0, greedy: bool = False,
                     seed: int = 0, batch: int = 256, max_new: int = MAX_NEW):
    """Returns per-question list of answer dicts {text, tokens, correct, valid, status, pred}."""
    out = [[] for _ in rows]
    g = torch.Generator().manual_seed(seed)
    jobs = [(i, s) for s in range(1 if greedy else n_samples) for i in range(len(rows))]
    enc = [encode_prompt(tok, r["question"]) for r in rows]
    t0 = time.perf_counter()
    for b in range(0, len(jobs), batch):
        chunk = jobs[b:b + batch]
        resp, _ = sample(model, [enc[i] for i, _ in chunk], max_new, tok.eos_id, tok.pad_id,
                         temperature=temperature, greedy=greedy, generator=g)
        for (i, _), r in zip(chunk, resp):
            text = decode_response(tok, r)
            gr = grade(text, rows[i]["answer"])
            out[i].append({"text": text, "tokens": len(r), "ended": bool(r and r[-1] == tok.eos_id), **gr})
    secs = time.perf_counter() - t0
    return out, secs / max(len(jobs), 1)


def summarize(rows, greedy, samples, ks=(1, 2, 4, 8, 16)) -> dict:
    s = {"n_questions": len(rows)}
    if greedy is not None:
        s["greedy_acc"] = float(np.mean([g[0]["correct"] for g in greedy]))
        s["greedy_format"] = float(np.mean([g[0]["valid"] for g in greedy]))
    if samples is not None:
        n = len(samples[0])
        corr = np.array([[a["correct"] for a in q] for q in samples], dtype=float)
        s["n_samples"] = n
        s["sample_acc"] = float(corr.mean())
        s["format_rate"] = float(np.mean([[a["valid"] for a in q] for q in samples]))
        for k in ks:
            if k <= n:
                s[f"pass@{k}"] = float(np.mean([pass_at_k(n, int(c), k) for c in corr.sum(1)]))
        s["median_tokens"] = float(statistics.median(a["tokens"] for q in samples for a in q))
        s["ended_rate"] = float(np.mean([[a["ended"] for a in q] for q in samples]))
        for key in ("family", "difficulty"):
            groups = {}
            for r, c in zip(rows, corr.mean(1)):
                groups.setdefault(r[key], []).append(c)
            s[f"sample_acc_{key}"] = {k: float(np.mean(v)) for k, v in sorted(groups.items())}
    if greedy is not None:
        for key in ("family", "difficulty"):
            groups = {}
            for r, g in zip(rows, greedy):
                groups.setdefault(r[key], []).append(g[0]["correct"])
            s[f"greedy_acc_{key}"] = {k: float(np.mean(v)) for k, v in sorted(groups.items())}
    return s


def evaluate(model, tok, rows, n_samples=16, temperature=1.0, do_greedy=True, seed=0, max_new=MAX_NEW):
    greedy = samples = None
    lat = {}
    if do_greedy:
        greedy, lat["greedy_s_per_answer_batched"] = generate_answers(model, tok, rows, greedy=True, max_new=max_new)
    if n_samples:
        samples, lat["sample_s_per_answer_batched"] = generate_answers(model, tok, rows, n_samples, temperature,
                                                                       seed=seed, max_new=max_new)
    s = summarize(rows, greedy, samples)
    s.update(lat)
    s["temperature"] = temperature
    s["max_new"] = max_new
    return s, greedy, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    model, _ = load_policy(ckpt_path(args.ckpt) if not args.ckpt.endswith(".pt") else args.ckpt)
    tok = load_tokenizer()
    rows = dataset.load(args.split)
    if args.limit:
        rows = rows[:args.limit]
    s, greedy, samples = evaluate(model, tok, rows, args.samples, args.temperature, not args.no_greedy, args.seed,
                                  args.max_new)
    s["ckpt"] = args.ckpt
    save_json(s, METRICS, "eval", args.split, f"{args.name}.json")
    recs = []
    for i, r in enumerate(rows):
        recs.append({"id": r["id"], "family": r["family"], "difficulty": r["difficulty"], "answer": r["answer"],
                     "greedy": greedy[i][0] if greedy else None, "samples": samples[i] if samples else None})
    write_jsonl(recs, GENERATIONS, args.split, f"{args.name}.jsonl")
    keys = ["greedy_acc", "sample_acc", "format_rate", "pass@1", "pass@4", "pass@16", "median_tokens"]
    print(args.name, {k: round(s[k], 4) for k in keys if k in s})


if __name__ == "__main__":
    main()
