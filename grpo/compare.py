"""Paired comparisons between systems on the same test questions (Phases 11, 12, 14).

    python -m grpo.compare --split test

Every system answered the same questions, so differences are paired by question. For each pair (A, B) and metric
(greedy accuracy, sampled accuracy = mean over 16 samples, pass@4, pass@16, format rate) the difference A - B and a
95% bootstrap interval (10,000 resamples of questions, fixed seed) are reported, plus the same per difficulty.
Pairs and hypotheses are fixed in configs/hypotheses.md before the test set is used.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import GENERATIONS, METRICS, load_config, read_jsonl, save_json
from .evaluate import pass_at_k


def per_question(recs, metric):
    out = []
    for r in recs:
        s = r["samples"]
        if metric == "greedy":
            out.append(float(r["greedy"]["correct"]))
        elif metric == "sample":
            out.append(float(np.mean([a["correct"] for a in s])))
        elif metric == "format":
            out.append(float(np.mean([a["valid"] for a in s])))
        elif metric.startswith("pass@"):
            k = int(metric[5:])
            out.append(pass_at_k(len(s), sum(a["correct"] for a in s), k))
        elif metric == "tokens":
            out.append(float(np.median([a["tokens"] for a in s])))
    return np.array(out)


def paired(a, b, n_boot=10_000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return {"a": float(a.mean()), "b": float(b.mean()), "diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


METRICS_LIST = ["greedy", "sample", "pass@4", "pass@16", "format"]


def compare(split, pairs):
    out = []
    cache = {}
    for a, b, question in pairs:
        for n in (a, b):
            if n not in cache:
                cache[n] = read_jsonl(GENERATIONS, split, f"{n}.jsonl")
        ra, rb = cache[a], cache[b]
        assert [r["id"] for r in ra] == [r["id"] for r in rb]
        entry = {"a": a, "b": b, "question": question}
        for m in METRICS_LIST:
            if m == "greedy" and (ra[0]["greedy"] is None or rb[0]["greedy"] is None):
                continue  # temperature-control systems have no greedy answers
            entry[m] = paired(per_question(ra, m), per_question(rb, m))
            for dif in ("easy", "multi"):
                sel = np.array([r["difficulty"] == dif for r in ra])
                entry[f"{m}_{dif}"] = paired(per_question(ra, m)[sel], per_question(rb, m)[sel])
        out.append(entry)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs", default="systems.yaml")
    args = ap.parse_args()
    cfg = load_config(args.pairs)
    res = compare(args.split, [tuple(p) for p in cfg["pairs"]])
    save_json(res, METRICS, f"{args.split}_comparisons.json")
    for e in res:
        print(f"{e['a']:>22} vs {e['b']:<22} " + "  ".join(
            f"{m} {100 * e[m]['diff']:+.1f} [{100 * e[m]['ci95'][0]:+.1f},{100 * e[m]['ci95'][1]:+.1f}]"
            for m in ("greedy", "sample", "pass@16") if m in e))


if __name__ == "__main__":
    main()
