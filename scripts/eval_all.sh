#!/usr/bin/env bash
# Phase 12: every system on the 1,000 test problems (1 greedy + 16 samples at T = 1.0), same script, same seed.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
# sft_mid (pre-registered rule): the SFT checkpoint whose SFT seconds since step 1,000 are closest to grpo_early_s0's
MID=$($PY - <<'PY'
import csv, grpo
g = grpo.load_json(grpo.METRICS, "train", "grpo_early_s0", "summary.json")["train_seconds"]
rows = [r for r in csv.DictReader(open(grpo.os.path.join(grpo.METRICS, "train", "sft", "log.csv"))) if r["val_greedy_acc"]]
t = {int(r["step"]): float(r["secs"]) for r in rows}
best = min((s for s in t if s >= 1000), key=lambda s: abs(t[s] - t[1000] - g))
grpo.save_json({"grpo_early_s0_train_seconds": g, "sft_seconds_at_1000": t[1000], "chosen_step": best,
                "sft_seconds_since_1000": t[best] - t[1000]}, grpo.METRICS, "sft_mid_choice.json")
print(f"sft/step_{best:05d}")
PY
)
echo "sft_mid = $MID"
ev(){ [ -f results/metrics/eval/test/$1.json ] || $PY -m grpo.evaluate --ckpt "$2" --name "$1" --split test --samples 16 ${3:-}; }
ev base base
ev sft_1000 sft/step_01000
ev sft sft
ev sft_mid "$MID"
ev sft_extra sft_extra
for r in grpo_s0 grpo_s1 grpo_s2 grpo_early_s0 grpo_early_s1 grpo_early_s2 early_nofmt early_g2 early_g4 early_beta0; do ev $r $r; done
# temperature control: is a GRPO gain more than what a lower sampling temperature gives the SFT model for free?
ev sft_1000_t07 sft/step_01000 "--temperature 0.7 --no-greedy"
ev sft_t07 sft "--temperature 0.7 --no-greedy"
echo "eval done"
