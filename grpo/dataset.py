"""Build and load the fixed dataset files (Phase 2).

    python -m grpo.dataset        -> data/{train,val,test}.jsonl + results/metrics/dataset_stats.json

train 100,000 / val 500 / test 1,000 unique problems, natural family mix (20/15/30/15/20 %), fixed seeds.
The split of a problem is decided by a hash of its computation chain (grpo.tasks.split_of), so no computation is
shared between splits; this script also checks that no question text or chain overlaps.
"""
from __future__ import annotations

import collections
import os

from . import DATA, METRICS, read_jsonl, save_json, write_jsonl
from .tasks import Problem, build_split

SIZES = {"train": 100_000, "val": 500, "test": 1_000}
SEEDS = {"train": 1, "val": 2, "test": 3}


def load(split: str) -> list[dict]:
    return read_jsonl(DATA, f"{split}.jsonl")


def stats(rows: list[dict]) -> dict:
    fam = collections.Counter(r["family"] for r in rows)
    dif = collections.Counter(r["difficulty"] for r in rows)
    ans = sorted(r["answer"] for r in rows)
    return {"n": len(rows), "family": dict(fam), "difficulty": dict(dif),
            "answer_median": ans[len(ans) // 2], "answer_max": ans[-1]}


def main():
    out = {}
    splits = {}
    for s, n in SIZES.items():
        rows = [p.to_dict() for p in build_split(s, n, SEEDS[s])]
        write_jsonl(rows, DATA, f"{s}.jsonl")
        splits[s] = rows
        out[s] = stats(rows)
        print(s, out[s])
    for a in SIZES:
        for b in SIZES:
            if a < b:
                qa = {r["question"] for r in splits[a]}
                ca = {r["chain"] for r in splits[a]}
                ov_q = sum(r["question"] in qa for r in splits[b])
                ov_c = sum(r["chain"] in ca for r in splits[b])
                out[f"overlap_{a}_{b}"] = {"question": ov_q, "chain": ov_c}
                assert ov_q == 0 and ov_c == 0, (a, b, ov_q, ov_c)
    save_json(out, METRICS, "dataset_stats.json")
    print("no question or chain overlap between splits")


if __name__ == "__main__":
    main()
