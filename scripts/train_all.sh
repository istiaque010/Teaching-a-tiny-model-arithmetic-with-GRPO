#!/usr/bin/env bash
# Phases 10 and 14: all GRPO runs, one after another (CPU), then the matched "more SFT" control.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
for r in grpo_early_s0 grpo_early_s1 grpo_early_s2 early_nofmt early_g2 early_g4 early_beta0 grpo_s0 grpo_s1 grpo_s2; do
  [ -f checkpoints/$r/final.pt ] || $PY -m grpo.train_grpo --config $r.yaml > results/metrics/logs/$r.log 2>&1
  echo "$r done $(date +%H:%M)"
done
# sft_extra: same wall-clock as grpo_s0's training, at SFT's measured seconds per update
$PY - <<'PY'
import json, grpo
g = grpo.load_json(grpo.METRICS, "train", "grpo_s0", "summary.json")["train_seconds"]
s = grpo.load_json(grpo.METRICS, "train", "sft", "summary.json")
steps = round(g / (s["train_seconds"] / s["config"]["steps"]))
open("configs/sft_extra.yaml", "w").write(f"""# Matched control (Phase 11): more SFT from the final SFT model for the same training wall-clock as grpo_s0
# ({g:.0f} s / {s['train_seconds'] / s['config']['steps']:.3f} s per update = {steps} updates), constant lr 3e-5 (= the end of SFT's schedule).
run: sft_extra
init: sft
seed: 1
steps: {steps}
batch: 64
lr: 3.0e-5
min_lr: 3.0e-5
schedule: constant
warmup: 0
weight_decay: 0.1
eval_every: {max(50, steps // 4)}
eval_samples: 4
""")
print("sft_extra steps", steps)
PY
[ -f checkpoints/sft_extra/final.pt ] || $PY -m grpo.train_sft --config sft_extra.yaml > results/metrics/logs/sft_extra.log 2>&1
echo "all training done $(date +%H:%M)"
