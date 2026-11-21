"""Error analysis and failure modes (Phase 15), fully automatic and deterministic.

    python -m grpo.errors --split test --systems sft grpo_s0 ...

Every response in this format can be checked step by step, so instead of judging explanations by eye, each
response is compared with the problem's true computation chain (e.g. "3*4=12;12-5=7"):

  invalid          no parseable final answer (sub-type = parser status)
  misread          a step uses a number that is neither in the question nor an earlier result
  arithmetic       a step's own result is wrong (e.g. "88 - 58 = 20")
  wrong_plan       every step is arithmetically right and uses known numbers, but the computation differs from the
                   true chain (wrong operation, wrong order, a step missing or extra)
  final_mismatch   the steps reach the right result but the tagged final answer is a different number
  other            anything else (e.g. no steps written)

and for *correct* final answers, whether the steps are faithful:

  faithful         the steps equal the true chain (up to commutative order)
  lucky            correct final answer, but some step is wrong or the steps are not the true chain
                   (the guide's "reasoning mismatch": the answer is right for the wrong reason)

Also: distinct responses among a question's 16 samples (mode collapse), and response-length spread.
"""
from __future__ import annotations

import argparse
import collections
import re

import numpy as np

from . import GENERATIONS, METRICS, read_jsonl, save_json
from . import dataset
from .tasks import _canon

STEP = re.compile(r"(\d+) ([+\-*]) (\d+) = (\d+)\.")


def steps_of(text):
    body = text.split("Final answer:")[0]
    return [(int(a), o, int(b), int(c)) for a, o, b, c in STEP.findall(body)]


def classify(text, prob, ans):
    """prob: dataset row; ans: grade dict for this response. Returns a category string."""
    st = steps_of(text)
    true_chain = prob["chain"]
    canon = ";".join(_canon(a, o, b) for a, o, b, c in st)
    arith_ok = all({"+": a + b, "-": a - b, "*": a * b}[o] == c for a, o, b, c in st)
    if ans["correct"]:
        return "faithful" if (arith_ok and canon == true_chain) else "lucky"
    if not ans["valid"]:
        return "invalid:" + ans["status"]
    if not st:
        return "other"
    qnums = {int(x) for x in re.findall(r"\d+", prob["question"])}
    known = set(qnums)
    for a, o, b, c in st:
        if a not in known or b not in known:
            return "misread"
        known.add(c)
    if not arith_ok:
        return "arithmetic"
    if st[-1][3] == prob["answer"] and ans["pred"] != prob["answer"]:
        return "final_mismatch"
    if canon != true_chain:
        return "wrong_plan"
    return "other"


def analyse(split, name, probs):
    recs = read_jsonl(GENERATIONS, split, f"{name}.jsonl")
    cats = collections.Counter()
    by_fam = collections.defaultdict(collections.Counter)
    distinct, lens, n = [], [], 0
    examples = collections.defaultdict(list)
    for r in recs:
        p = probs[r["id"]]
        for a in r["samples"]:
            c = classify(a["text"], p, a)
            cats[c] += 1
            by_fam[p["family"]][c] += 1
            lens.append(a["tokens"])
            n += 1
            if len(examples[c]) < 6:
                examples[c].append({"id": r["id"], "question": p["question"], "answer": p["answer"],
                                    "response": a["text"]})
        distinct.append(len({a["text"] for a in r["samples"]}))
    wrong = sum(v for k, v in cats.items() if k not in ("faithful", "lucky"))
    correct = cats["faithful"] + cats["lucky"]
    return {"n_responses": n, "counts": dict(cats.most_common()),
            "share_of_wrong": {k: v / max(1, wrong) for k, v in cats.items() if k not in ("faithful", "lucky")},
            "lucky_share_of_correct": cats["lucky"] / max(1, correct),
            "by_family": {f: dict(c) for f, c in sorted(by_fam.items())},
            "distinct_per_16_mean": float(np.mean(distinct)),
            "single_answer_questions": float(np.mean([d == 1 for d in distinct])),
            "len_mean": float(np.mean(lens)), "len_sd": float(np.std(lens)), "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--systems", nargs="+", required=True)
    args = ap.parse_args()
    probs = {r["id"]: r for r in dataset.load(args.split)}
    out = {n: analyse(args.split, n, probs) for n in args.systems}
    save_json(out, METRICS, f"errors_{args.split}.json")
    for n, s in out.items():
        top = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(s["share_of_wrong"].items(), key=lambda x: -x[1])[:4])
        print(f"{n:>16}: lucky {100 * s['lucky_share_of_correct']:.2f}% of correct | wrong: {top} | "
              f"distinct/16 {s['distinct_per_16_mean']:.2f}")


if __name__ == "__main__":
    main()
