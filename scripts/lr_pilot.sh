#!/usr/bin/env bash
# Phase 10 learning-rate pilot: three GRPO runs of 60 iterations, validation only.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for lr in 1e5 3e5 1e4; do
  .venv/Scripts/python -m grpo.train_grpo --config lr_$lr.yaml > results/metrics/logs/lr_$lr.log 2>&1
done
echo lr pilot done
