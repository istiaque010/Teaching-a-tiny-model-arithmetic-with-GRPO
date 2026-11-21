# Pre-registered hypotheses (written 2026-11-18, before any test-set evaluation and before the main GRPO runs)

Test set: 1,000 held-out problems (`data/test.jsonl`), each answered once greedily and 16 times by sampling at
temperature 1.0. "Sampled accuracy" = mean correctness over the 16 samples. Differences are paired by question with a
95% bootstrap interval (10,000 resamples). A hypothesis "holds" only if its interval is on the stated side.

| # | Hypothesis | Test |
|---|---|---|
| H1 | The Project 1 model cannot do the task before post-training: valid-format rate < 5% and accuracy < 2% | base on test |
| H2 | The SFT warm-up makes GRPO possible: SFT sampled accuracy ≥ 30% and ≥ 50% of training groups (G = 8) have reward variance | sft on test; GRPO log at iteration 0 |
| H3 | **Main.** GRPO improves on its starting point: grpo sampled accuracy − sft sampled accuracy > 0 | grpo vs sft |
| H4 | **Main.** GRPO beats more SFT at the same training compute (wall-clock): grpo sampled accuracy > sft_extra | grpo vs sft_extra |
| H5 | The gain is not only temperature sharpening: grpo greedy accuracy > sft greedy accuracy | grpo vs sft, greedy |
| H6 | GRPO does not widen coverage: grpo pass@16 − sft pass@16 < +1 point (RL concentrates mass on answers the SFT model can already find) | grpo vs sft, pass@16 |
| H7 | The format bonus is not needed once SFT gives ~100% valid format: \|grpo − grpo_nofmt\| sampled accuracy < 2 points | ablation |
| H8 | Larger groups help at the same number of responses: grpo (G = 8) sampled accuracy > grpo_g2 (G = 2) | ablation |
| H9 | The KL penalty limits drift: KL to the SFT policy at the end of training is lower with β = 0.04 than β = 0, at a cost of < 2 points accuracy | ablation (training log, test) |

Main GRPO = 3 seeds; H3–H6 use seed 0 for the paired test and report the spread of the 3 seeds alongside.

## Addendum (2026-11-18, after watching SFT *validation* accuracy, before any GRPO run and before any test use)

The main SFT run reached 96% greedy / 90% sampled validation accuracy by step 2,250 of 3,000, so GRPO from the final
SFT model may have little reward variance (most groups all-correct). The plan above is kept unchanged, and one
second starting point is added so the study also covers a regime with plenty of signal:

* **Early start.** `grpo_early` starts from the SFT checkpoint at step 1,000 (≈55% sampled validation accuracy).
  Its matched "more SFT" control is the *same* SFT run continued for the same training wall-clock time from that
  checkpoint: the SFT checkpoint whose cumulative SFT seconds − 750 s (the time at step 1,000) is closest to the
  GRPO run's training seconds (`sft_mid`, chosen by that rule only).

| # | Hypothesis | Test |
|---|---|---|
| H10 | From the early start, GRPO improves on its start: grpo_early − sft_1000 sampled accuracy > 0 | test |
| H11 | From the early start, GRPO beats more SFT at matched wall-clock: grpo_early > sft_mid, sampled accuracy | test |

**Ablations move to the early start (written while the learning-rate pilot runs, before any of its results were
read).** The final SFT model reaches 98.6% greedy / 96.9% sampled validation accuracy, so from it almost every group
would be all-correct and an ablation could not show anything. H7–H9 are therefore tested from the early start:
`grpo_early` (G = 8, β = 0.04, format bonus 0.1) against `early_nofmt` (H7), `early_g2` (H8; G = 2 with 4× the
prompts per step, same 128 responses per step) and `early_beta0` (H9). `early_g4` is reported as a third point for H8.
The main-start runs (`grpo` and `sft_extra`, H3–H6) are kept exactly as planned.
