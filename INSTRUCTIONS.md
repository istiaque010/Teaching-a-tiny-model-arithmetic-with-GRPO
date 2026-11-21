# Project 8 — GRPO Post-Training of a Small Language Model: phase-by-phase instructions

Source plan: `docs/Project_8_GRPO_Post_Training_Small_Language_Model_Guide.docx` (kept locally, not in the repository).

**Goal:** post-train Project 1's small autoregressive language model with Group Relative Policy Optimization (GRPO)
on simple arithmetic whose answers can be checked automatically, and measure whether reward-based training improves
correctness compared with the original model and with supervised fine-tuning (SFT) alone.

**Research question:** under a fixed compute budget, can GRPO improve a small model's answer accuracy relative to
supervised fine-tuning and the original policy?

```
Project 1 GPT (13.9M, TinyStories only) → baseline (0%) → SFT warm-up on worked solutions
   → GRPO: 8 sampled answers per question → 0/1 correctness (+0.1 format) → group-relative advantages
   → clipped policy update (+ KL to the SFT model) → held-out accuracy, pass@k, cost, ablations, error analysis
```

## How to use this file
1. Work through the phases in order. Each has **Learn**, **Files**, **Run**, **Check** and **Result**.
2. Run everything from the `project8` folder with `.venv` active (Windows: `.venv\Scripts\activate`).
3. Status changes from ⬜ to ✅ when the phase is done and checked.

## Progress

| # | Phase | Status |
|---|---|---|
| 1 | Prerequisites and environment | ✅ |
| 2 | A narrow verifiable task | ✅ |
| 3 | Prompt and answer format, parser | ✅ |
| 4 | Baseline of the original model | ✅ |
| 5 | SFT warm-up | ✅ |
| 6 | Reward functions | ✅ |
| 7 | Grouped sampling | ✅ |
| 8 | Group-relative advantages | ✅ |
| 9 | GRPO objective | ✅ |
| 10 | Training with GRPO | ✅ |
| 11 | Comparison baselines | ✅ |
| 12 | Reasoning performance | ✅ |
| 13 | Efficiency and compute | ✅ |
| 14 | Ablations | ✅ |
| 15 | Error analysis | ✅ |
| 16 | Repository and reproducibility | ✅ |

## Decisions for this PC and this project (senior-researcher choices, with reasons)

- **CPU, not GPU** (no GPU on this PC), fp32, 24 threads, as in Projects 1–7. "GPU-hours" and "peak VRAM" become CPU
  wall-clock and peak process memory. TRL is not used: it targets Hugging Face models on GPUs, and the guide asks to
  implement the GRPO loss by hand for learning. Every formula is in `grpo/grpo_loss.py` and tested.
- **Starting model = Project 1's `small` checkpoint** (13.9M parameters, 6 layers, trained only on TinyStories),
  sha256-checked, read-only. The guide warns such a model may lack arithmetic; Phase 4 confirms it (0% valid
  answers), so an SFT warm-up is required (Phase 5). No other pretrained model is used.
- **Synthetic arithmetic, five families** (one- and two-step sums, products and short word problems), generated
  from integers so every answer is exact. Splits are decided by a hash of the *computation*, so no computation
  appears in two splits even with different wording.
- **Strict parser.** One fixed format ending in `<answer>N</answer>`; conflicting tags, non-numbers and unclosed tags
  are invalid, so the reward cannot be collected by odd outputs.
- **A fast sampler of our own.** Project 1's generator recomputes the whole sequence for every token. A key/value
  cache over the same weights samples 128 answers in ~1.5 s, which makes 8 answers per question affordable on a CPU.
  It is tested to give exactly Project 1's tokens and probabilities.
- **No dropout in post-training,** so the model that samples and the model being optimised are the same function and
  the probability ratio is exactly 1 before an update.
- **Sampling = the policy itself:** temperature 1.0, no top-k / top-p, for rollouts and for the sampled evaluation.
  Greedy answers are measured separately.
- **Two starting points (decided on validation, before any GRPO run or test use; see `configs/hypotheses.md`).**
  The SFT model reaches 98.6% greedy validation accuracy, which leaves GRPO almost nothing to learn from. The
  planned comparison from that final SFT model is kept, and a second start, the SFT checkpoint at step 1,000
  (~55% sampled accuracy), is added. The ablations run there, where rewards actually vary.
- **Fair "more SFT" control at the same wall-clock.** GRPO is compared with continuing SFT for the same training
  time, from the same starting checkpoint. Without it, a GRPO gain could just be "more training".
- **Tune on validation, test once.** Learning rate from a validation pilot. Hypotheses written before the test set
  is used; every system answers the same 1,000 test questions (1 greedy + 16 samples).

---
## Phase 1 — Prerequisites and environment

**Learn:** a causal language model gives, at every position, a probability for each possible next token. Generation
= repeatedly sample (or take the most likely, *greedy*) next token. The **log-probability** of a response is the sum
of the log-probabilities of its tokens. That is the quantity reinforcement learning pushes up for good responses
and down for bad ones. **Cross-entropy** (SFT's loss) is minus the log-probability of the correct next token.
A **policy** is just the model seen as a rule that picks actions (tokens). A **KL penalty** measures how far the new
policy's probabilities have moved from a reference policy and charges for it.

**Files:** `requirements.txt`, `requirements.lock.txt`, `grpo/__init__.py` (paths; makes Project 1's `src` importable
read-only), `grpo/policy.py` (loading, padded forward, cached sampling, token log-probabilities),
`scripts/env_report.py`

**Run:**
```
python -m venv .venv
.venv\Scripts\pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
python scripts/env_report.py          # -> results/metrics/env.json
python -m pytest                      # all unit tests
```

**Check:** the model loads, generates, and token log-probabilities can be computed; the padded and cached paths give
Project 1's numbers (`tests/test_policy.py`).

**Result:** Python 3.12.10, torch 2.14.1+cpu, 24 CPU threads, 63.7 GB RAM, no CUDA. Base checkpoint
`project1/checkpoints/small/model_best.pt`, sha256 `b4a9986a…611361`, 13,891,584 parameters (6 layers × 6 heads,
width 384, 256-token context, 8,192-token vocabulary). Against Project 1's own forward pass the padded forward
matches to 2·10⁻⁵, cached greedy decoding gives identical tokens, and rollout log-probabilities match a recompute to
5·10⁻⁶. Sampling 128 answers of up to 64 tokens takes 1.45 s (≈5,400 tokens/s).

---
## Phase 2 — A narrow verifiable task

**Learn:** reinforcement learning needs a reward. The cleanest reward is one a program can check with no judgement:
"is the final number right?". Synthetic arithmetic gives exact answers, a difficulty tag per problem, and as many
problems as needed. The split must be leak-free: if "3 × 4 = 12" is in training, the test set must not contain it,
even reworded as "3 bags with 4 apples".

**Files:** `grpo/tasks.py` (generators, hash split), `grpo/dataset.py` (fixed files + overlap check),
`tests/test_tasks.py`

| Family | Steps | Tag | Example |
|---|---|---|---|
| add | 1 | easy | `47 + 38 = ?` (numbers 1–99, results ≥ 0) |
| mul | 1 | easy | `7 * 16 = ?` (one factor 2–12, the other 2–99) |
| multi | 2 | multi | `(12 + 8) * 3 = ?`, `4 * 92 + 92 = ?`, `97 + 51 - 77 = ?` |
| word1 | 1 | easy | `Mia has 3 bags with 4 apples in each. How many apples does Mia have?` |
| word2 | 2 | multi | `Ben has 7 baskets with 7 cookies in each. He gives 18 cookies to Max. How many cookies does Ben have now?` |

**Run:** `python -m grpo.dataset` → `data/{train,val,test}.jsonl`, `results/metrics/dataset_stats.json`

**Check:** every gold solution re-computes to its answer; the split of a problem is a function of its canonical chain
(commutative steps sorted, so `7*16` and `16*7` share a split); no question text or chain is shared between splits.

**Result:** 100,000 train / 500 validation / 1,000 test problems, zero question or chain overlap (asserted). Test:
add 201, mul 105, multi 347, word1 170, word2 177 (476 easy, 524 multi-step); median answer 68, largest 1,189.
The one-step families have a small space of distinct computations (e.g. ~2,000 products), so the training set holds
fewer of them (mul 1,797, add 11,329). Hash buckets 0–87 / 88–91 / 92–99 give train / validation / test.

---
## Phase 3 — Prompt and answer format, parser

**Learn:** the model must put its answer where a program can find it. One format for the whole project:

```
<BOS>Question: Ben has 7 baskets with 7 cookies in each. He gives 18 cookies to Max. How many cookies does Ben have now?
Solution: 7 * 7 = 49. 49 - 18 = 31.
Final answer: <answer>31</answer><EOS>
```

The *prompt* ends after `Solution:`; everything after it is the *response*. The parser is strict on purpose: a reward
that can be earned by a trick will be found by the optimiser.

**Files:** `grpo/prompts.py`, `grpo/answer_parser.py`, `tests/test_rewards.py`

**Check:** valid (`12`, ` 12 `, `+12`, `12.0`, `1,234`, agreeing duplicates) and invalid / adversarial outputs
(empty, no tag, unclosed tag, `twelve`, `12 apples`, `12.5`, `1e3`, empty tag, bad commas `12,34`, nested tags,
conflicting tags, > 9 digits) are all classified correctly; prompt + response tokenise exactly like the joined string.

**Result:** all parser tests pass. The tokenizer (Project 1's byte-level BPE) writes numbers mostly digit by digit
(`Ġ2`, `3`), has no unknown tokens on this task, and needs ~25 prompt tokens and ~40 response tokens per problem.

---
## Phase 4 — Baseline of the original model

**Learn:** measure before training, so any later gain has a reference. If the model never produces a parseable
answer, every reward is 0, every group has the same reward, and GRPO has nothing to learn from.

**Run:** `python -m grpo.evaluate --ckpt base --name base --split val --samples 4` (test: Phase 12)

**Result:** validation: greedy accuracy **0.0%**, sampled accuracy 0.0%, valid format **0.0%**. Every answer runs to
the 64-token limit with story text ("... Mia has three oranges. Mia likes orange. They smell good. One day…").
The model has never seen `<answer>` tags or worked sums: a TinyStories-only model needs a warm-up.

---
## Phase 5 — SFT warm-up

**Learn:** supervised fine-tuning shows the model correct worked solutions and trains it with cross-entropy on the
response tokens only (the question is given, not learned). It teaches the format and the arithmetic. The SFT model
is both GRPO's starting point and a baseline every GRPO result must beat.

**Files:** `grpo/train_sft.py`, `configs/sft_pilot.yaml`, `configs/sft.yaml`

**Run:** `python -m grpo.train_sft --config sft.yaml` (3,000 updates × 64 examples, lr 3e-4 → 3e-5 cosine, 100
warm-up updates, AdamW, weight decay 0.1, gradient clip 1.0)

**Check:** validation accuracy every 250 updates, a checkpoint each time; validation and test questions are never in
SFT data (different splits).

**Result:** validation accuracy on 500 held-out problems (greedy / sampled at T = 1.0):

| SFT update | 0 (Project 1 model) | 1,000 | 1,500 | 2,000 | 2,250 | 2,500 | 3,000 (final) |
|---|---|---|---|---|---|---|---|
| greedy | 0% | 64.4% | 79.6% | 89.6% | 96.0% | 97.4% | **98.6%** |
| sampled | 0% | 54.8% | 72.1% | 84.8% | 89.9% | 94.0% | **96.9%** |
| seconds | 0 | 750 | 1,123 | 1,505 | 1,691 | 1,885 | 2,261 |

Valid format is ~100% from update 200 on. 3,000 updates x 64 = 192,000 examples (7.7M response tokens), 38 minutes
on this CPU (0.75 s per update). The final SFT model is already near the ceiling, which is why an *early start*
(update 1,000) was added before any GRPO run (pre-registered addendum in `configs/hypotheses.md`).

---
## Phase 6 — Reward functions

**Learn:** `R = 1.0 × correctness + 0.1 × format`. Correctness is 1 if the parsed final answer equals the true
answer. Format is 1 if there is exactly one parseable final answer. The format bonus is small so that a well-formed
wrong answer (0.1) is always worth far less than a right one (1.1). Length is never rewarded: a long explanation is not
evidence of correct reasoning.

**Files:** `grpo/rewards.py`, `tests/test_rewards.py`

**Check:** correct → 1.1, well-formed wrong → 0.1, malformed / empty / conflicting → 0, `w_format = 0` → 1.0 / 0;
the same answer with or without 20 extra sentences gets the same reward.

**Result:** all reward tests pass; the three components (total, correctness, format) are logged separately at every
GRPO update, and the format bonus is ablated in Phase 14.

---
## Phase 7 — Grouped sampling

**Learn:** GRPO asks the model the same question G times (G = 8 here) and compares the answers *with each other*.
For every sampled token the sampler also records its log-probability under the policy that produced it (the *old*
policy), because the update later measures how far the new policy has moved from it. Only response tokens enter the
loss: the prompt is masked out.

**Files:** `grpo/policy.py` (`sample`, `pack`, `token_logprobs`), `tests/test_policy.py`

**Check:** responses stop at `<EOS>`; the response mask covers exactly the response tokens; recorded log-probabilities
equal a fresh recompute at temperature 1.0 and 0.7.

**Result:** one rollout batch = 16 questions × 8 answers = 128 responses, sampled in ~1.7 s on this CPU. All
policy tests pass.

---
## Phase 8 — Group-relative advantages

**Learn:** the *advantage* says how much better a response was than expected. GRPO's "expected" is simply the average
of its own group: `A_i = (r_i − mean(r)) / (std(r) + ε)`. Rewards [1, 0, 1, 0] → mean 0.5 → advantages +1, −1, +1, −1.
No separate value network is needed (the big saving over PPO). If all 8 answers are equally right (or wrong), every
advantage is 0 and the question teaches nothing. The share of such *zero-variance groups* is the key health number.

**Files:** `grpo/advantages.py`, `tests/test_advantages.py`

**Check:** the guide's example gives ±1; equal-reward groups give exactly 0 with no division by zero; within a group
advantages have mean 0 and standard deviation 1; order is preserved (1.1 > 1.0 > 0.1 > 0).

**Result:** all tests pass. The population standard deviation (divide by G) is used.

---
## Phase 9 — GRPO objective

**Learn:** for every response token, `ratio = exp(log π_new − log π_old)`. The update maximises
`min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`: it raises good responses and lowers bad ones, but stops pushing a token once
its probability has moved more than ε = 20% in the advantage's direction (*clipping*), so one batch cannot change the
policy too much. A KL penalty `β·KL(π ‖ π_ref)` to the frozen starting model (β = 0.04) adds a further brake.

**Variant used (implementations differ):**

| Choice | This project |
|---|---|
| KL estimator | "k3": `exp(ref − new) − (ref − new) − 1` per token (≥ 0, unbiased) |
| Averaging | token-level: sum over all response tokens of the minibatch / their number |
| Advantage | group mean and population std; zero-variance groups skipped (their advantage is 0) |
| Old log-probs | recorded at sampling time and fixed for both passes over the rollout batch |
| Passes | 2 passes × 2 minibatches of 64 responses = 4 updates per rollout batch |
| Optimiser | AdamW, β = (0.9, 0.99), no weight decay, gradient clip 1.0, 5 warm-up iterations |

**Files:** `grpo/grpo_loss.py`, `tests/test_grpo_loss.py`

**Check:** on-policy (ratio = 1) the loss equals −mean advantage; the gradient raises good and lowers bad responses
and is zero for advantage-0 and masked tokens; a token with ratio 1.5 and A > 0 gets no gradient (clipped), one with
ratio 0.5 still does; KL is 0 at the reference and positive away from it; the loss stays finite at extreme values.

**Result:** all loss tests pass. A first optimiser step on real rollouts (pilot, `configs/grpo_pilot.yaml`) gave a
finite loss, gradient norm ~0.5, clip fraction 0.3%.

---
## Phase 10 — Training with GRPO

**Learn:** the loop is sample → score → advantages → 4 clipped updates → repeat. Logged every iteration: reward and
its parts, zero-variance share, response length, entropy, KL to the reference, clip fraction, approximate KL between
old and new policy, gradient norm, rollout and optimiser seconds, memory.

**Files:** `grpo/train_grpo.py`, `configs/lr_*.yaml`, `configs/grpo_s*.yaml`, `configs/grpo_early_s*.yaml`,
`scripts/lr_pilot.sh`, `scripts/train_all.sh`

**Run:** `bash scripts/lr_pilot.sh` then `bash scripts/train_all.sh`

**Learning-rate pilot (validation only, 60 iterations from the step-1,000 SFT checkpoint, seed 100):**

| lr | val greedy, start → 60 | val sampled, start → 60 |
|---|---|---|
| **1e-5** | 64.4% → **72.6%** | 53.4% → **63.3%** |
| 3e-5 | 64.4% → 67.6% | 53.4% → 60.3% |
| 1e-4 | 64.4% → 45.0% | 53.4% → 36.5% (collapse) |

1e-5 chosen (highest validation sampled accuracy at the end). 1e-4 breaks the policy within 20 iterations: the
learning rate, not the clip, is what keeps GRPO stable here.

**Result:** all 10 GRPO runs finished, 300 iterations x 128 responses = 38,400 responses each (about 1.53M response
tokens). Last 20 iterations vs first 10 (`results/metrics/train/<run>/log.csv`, figure 2):

| run | start | zero-variance groups, first 10 → last 20 | reward | val sampled, start → end | train time | peak RAM |
|---|---|---|---|---|---|---|
| grpo_s0 / s1 / s2 | final SFT | 87–90% → 89–91% | 1.07 → 1.07 | 96.9% → 97.5 / 97.2 / 97.3% | 10.3 / 10.3 / 9.4 min | 2.5–2.7 GB |
| grpo_early_s0 / s1 / s2 | SFT 1,000 | 37–47% → 54–55% | 0.70 → 0.84 | 53.5% → 70.2 / 71.8 / 67.7% | 16.9 / 17.0 / 17.5 min | 3.5–3.8 GB |

What this shows:
* **From the final SFT model GRPO has almost nothing to learn from.** About 89% of groups are all-correct (or
  all-wrong), so their advantages are zero; reward stays flat at 1.07 for 300 iterations.
* **From the early start the loop works as intended.** About half the groups carry signal, reward rises 0.70 → 0.84
  and validation sampled accuracy rises 54% → 68–72%, but the curve flattens after about 100 iterations (figure 3).
  As the model gets better, more groups become all-correct and the signal shrinks again.
* Stable: clip fraction about 0.1%, approximate KL between old and new policy about 0.0005, KL to the reference
  0.02–0.04, entropy falls slightly (0.035 → 0.026–0.033 nats/token), response length unchanged (about 40 tokens).
  No collapse at lr 1e-5.

---
## Phase 11 — Comparison baselines

**Learn:** a GRPO number means nothing alone. Every system answers the same 1,000 test problems with the same
script (`grpo/evaluate.py`: 1 greedy answer + 16 samples at T = 1.0, seed 0), and differences are *paired by
question* with a 95% bootstrap interval (`grpo/compare.py`). Baselines, all fixed in `configs/systems.yaml` before the
test set was used:

| system | what it is | why it is needed |
|---|---|---|
| base | Project 1 model, no post-training | does post-training do anything at all? |
| sft_1000, sft | SFT at update 1,000 and the final SFT model | the two GRPO starting points |
| **sft_mid** | the *same* SFT run continued from update 1,000 for the same wall-clock time GRPO used (update 2,250: 941 s vs GRPO's 1,012 s) | matched-compute control for the early start |
| **sft_extra** | the final SFT model trained 819 more SFT updates (constant lr 3e-5), sized to grpo_s0's training time (it took 564 s vs 618 s, slightly *less* compute than GRPO) | matched-compute control for the main start |
| sft_1000_t07, sft_t07 | the SFT models sampled at T = 0.7 instead of 1.0 | is a GRPO gain just "sharpening" that a lower temperature gives for free? |

**Files:** `configs/systems.yaml`, `scripts/eval_all.sh`, `grpo/compare.py`

**Run:** `bash scripts/eval_all.sh` then `python -m grpo.compare --split test`

---
## Phase 12 — Reasoning performance

**Learn:** greedy accuracy = the model's single best guess. Sampled accuracy = the average over 16 samples at
T = 1.0 (what GRPO itself optimises). pass@k = the chance that at least one of k samples is right (coverage).

**Run:** `python -m grpo.compare --split test`, `python -m grpo.plots` (figures 4, 5)

**Test results (1,000 problems, `results/metrics/eval/test/`):**

| system | greedy | sampled | pass@4 | pass@16 | format |
|---|---|---|---|---|---|
| base (Project 1) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| sft_1000 | 63.6% | 53.6% | 76.5% | 90.3% | 99.9% |
| **grpo_early** (s0 / s1 / s2) | 76.1 / 75.6 / 73.6% | 70.1 / 68.7 / 68.8% | 84.5% (s0) | 91.7 / 92.4 / 93.1% | 100% |
| sft_mid (more SFT, same time) | **94.9%** | **91.5%** | 97.7% | 99.3% | 100% |
| sft_1000 at T = 0.7 | – | 58.9% | 75.8% | 87.4% | 100% |
| sft (final) | 98.7% | 96.9% | 99.4% | 99.6% | 100% |
| **grpo** (s0 / s1 / s2) | 99.3 / 98.2 / 98.2% | 97.4 / 96.7 / 96.8% | 99.6% (s0) | 100.0 / 99.8 / 99.6% | 100% |
| sft_extra (more SFT, same time) | 98.9% | **97.7%** | 99.6% | 99.9% | 100% |
| sft at T = 0.7 | – | **98.0%** | 99.4% | 99.6% | 100% |

**Paired differences (points, 95% CI), all three seeds** (`results/metrics/test_comparisons.json`,
`test_comparisons_seeds.json`):

| comparison | greedy | sampled | pass@16 |
|---|---|---|---|
| grpo_early − sft_1000 | +12.5 / +12.0 / +10.0 (all CIs > 0) | **+16.5 / +15.0 / +15.1** (all > 0) | +1.4 [−0.4, +3.2] / +2.1 / +2.8 |
| grpo_early − sft_mid | −18.8 / −19.3 / −21.3 | **−21.4 / −22.9 / −22.8** (all < 0) | −7.6 / −6.9 / −6.2 |
| grpo_early_s0 − sft_1000 at T = 0.7 | – | +11.2 [+9.5, +12.9] | +4.3 [+2.4, +6.2] |
| grpo − sft | +0.6 [+0.0, +1.2] / −0.5 / −0.5 | +0.5 [−0.0, +0.9] / −0.2 / −0.1 | +0.4 / +0.2 / +0.0 |
| grpo − sft_extra | +0.4 / −0.7 / −0.7 (all CIs include 0) | −0.4 [−1.1, +0.2] / **−1.0 / −1.0** (s1, s2 CIs < 0) | +0.1 / −0.1 / −0.3 |
| grpo_s0 − sft at T = 0.7 | – | −0.7 [−1.2, −0.2] | +0.4 [+0.1, +0.8] |

**Pre-registered hypotheses (`configs/hypotheses.md`):**

| # | hypothesis | verdict |
|---|---|---|
| H1 | base model cannot do the task | ✅ holds: 0% format, 0% accuracy (it continues the text as a story) |
| H2 | SFT makes GRPO possible (sampled ≥ 30% and ≥ 50% of groups with variance) | ❌ half fails: SFT reaches 96.9%, but only about 11% of groups have reward variance from the final SFT model (the early start has 53–63%) |
| H3 | **main:** GRPO improves on SFT | ❌ not supported: +0.5 [−0.0, +0.9] for seed 0, −0.2 and −0.1 for seeds 1 and 2 |
| H4 | **main:** GRPO beats more SFT at matched compute | ❌ fails: seed 0 −0.4 (not significant), seeds 1 and 2 −1.0 (CI < 0), even though sft_extra had 9% less time |
| H5 | gain is not just sharpening (greedy improves) | ❌ not supported: greedy +0.6 [+0.0, +1.2] for s0, −0.5 for s1 and s2 |
| H6 | GRPO does not widen coverage (pass@16 gain < 1 point) | ✅ holds (+0.0 to +0.4), but at 99.6% there is no room. From the early start the gain is +1.4 to +2.8 (CI > 0 for 2 of 3 seeds): a small real widening |
| H10 | from the early start, GRPO improves on its start | ✅ holds strongly: +15 to +16.5 points sampled, +10 to +12.5 greedy, every seed |
| H11 | from the early start, GRPO beats more SFT at matched compute | ❌ fails strongly: −21 to −23 points sampled, every seed |

**What it means.** GRPO works: from a weak start it lifts accuracy by 15 points, and that gain is more than a lower
sampling temperature gives (+11 over T = 0.7; greedy also improves by 10–12 points, so it is not only sharpening).
But on this task the *same* compute spent on more supervised training is far better. SFT sees a correct worked
solution for every example (a learning signal on every token), while GRPO only learns from groups where some samples
are right and some are wrong, with one number of reward per answer. When correct demonstrations are cheap,
imitation wins. From the final SFT model, GRPO changes nothing measurable; simply sampling at T = 0.7 is better.

Per family (sampled, `sample_acc_family` in each eval JSON): GRPO's early gains are largest on the word problems
(word2 62 → 87%) and smallest on multiplication (44 → 56%); 2-step problems gain as much as 1-step ones
(easy 64 → 77%, multi 45 → 64%).

---
## Phase 13 — Efficiency and compute

**Learn:** compare methods at equal cost, and report what each costs. All numbers are from this PC (24-thread
CPU, fp32, no GPU) and `results/metrics/benchmark.json`.

**Run:** `python -m grpo.benchmark --systems base sft_1000=sft/step_01000 sft grpo_s0 grpo_early_s0 sft_extra --runs ...`

**Training cost:**

| run | wall-clock | what it consumed | peak RAM |
|---|---|---|---|
| SFT 3,000 updates | 37.7 min | 192,000 examples, 7.7M response tokens | – |
| GRPO from final SFT (each seed) | 9.4–10.3 min | 38,400 sampled responses, 1.53M tokens | 2.5–2.7 GB |
| GRPO from SFT 1,000 (each seed) | 16.9–17.5 min | 38,400 sampled responses, 1.53M tokens | 3.5–3.8 GB |
| sft_extra | 9.4 min | 52,416 examples, 2.1M tokens | – |

A GRPO iteration costs about 2–3.5 s: roughly half generating 128 answers (one token at a time), half the 4 clipped
updates plus the reference forward pass. Per second of training, SFT processes about 3,400 response tokens and GRPO
1,500–2,500. SFT learns from every token, while GRPO learns only from responses with non-zero advantage (45–89% of
groups give nothing). The early-start runs are slower than the main-start ones because more groups have signal, so
more responses go through the update.

**Inference (batch 1 greedy on 100 test questions, 3 repeats):** every post-trained model is the same 13.9M network
writing the same ~39 tokens, so cost is identical: 64–71 ms per answer, about 1.7 ms per token, 95–100 answers/s at
batch 64. The base model is slower per answer (107 ms) only because it never stops and writes the full 64 tokens.
GRPO did not change answer length.

*Caveat:* this PC runs vendor fan/monitoring software (about 12% idle CPU); the in-script background-load check reads
about 45% because PyTorch's worker threads keep spinning just after generation. It affects all systems equally.

---
## Phase 14 — Ablations

**Learn:** an ablation changes one setting and keeps the rest. All ablations start from SFT update 1,000 (where
there is signal), use the same 38,400 responses, and are compared with the main early-start run.

**Run:** configs `early_nofmt`, `early_g2`, `early_g4`, `early_beta0` (in `scripts/train_all.sh`);
`python -m grpo.drift` for policy drift; figure 6

| run | change | test sampled | vs grpo_early s0 / s1 / s2 (points) | KL to SFT 1,000 (per token) |
|---|---|---|---|---|
| grpo_early (s0 / s1 / s2) | G = 8, β = 0.04, format 0.1 | 70.1 / 68.7 / 68.8% | – | 0.0167 / 0.0166 / 0.0168 |
| early_nofmt | no format bonus | 68.9% | −1.2 [−2.3, −0.1] / +0.3 / +0.2 | 0.0157 |
| early_g4 | G = 4, 32 prompts/step | 70.9% | +0.8 / **+2.2 / +2.1** | 0.0146 |
| early_g2 | G = 2, 64 prompts/step | 70.9% | +0.8 / **+2.3 / +2.2** | 0.0132 |
| early_beta0 | no KL penalty | 68.8% | −1.3 [−2.6, −0.1] / +0.1 / +0.0 | 0.0170 |
| (sft_mid, for scale) | more SFT | 91.5% | | 0.0327 |

Drift is measured after training, the same way for every run: KL(policy ‖ SFT 1,000) on the policy's own samples,
250 questions x 8 (`results/metrics/drift_early.json`, standard error about 0.0008). This is needed because the
β = 0 run has no reference model during training, so its log has no KL.

| # | hypothesis | verdict |
|---|---|---|
| H7 | format bonus not needed (difference < 2 points) | ✅ holds: −1.2, +0.3, +0.2. SFT already gives 100% valid format |
| H8 | larger groups help (G = 8 > G = 2) | ❌ the opposite: G = 2 and G = 4 are 0.8–2.3 points *better* at the same number of responses |
| H9 | the KL penalty limits drift at < 2 points cost | ❌ not supported: drift is the same with and without it (0.0170 vs 0.0167) |

**Lessons.** (1) **Seeds matter as much as the ablation.** Compared with seed 0 alone, removing the format bonus or the
KL penalty looks "significantly worse" (CI excludes 0); compared with seeds 1 and 2 the difference is zero. The three
identical GRPO runs span 1.4 points, as much as most ablation effects; a one-seed ablation would have reported a
false finding. (2) With 128 responses per step, spreading them over more questions (64 x 2) beats more samples per
question (16 x 8): more different problems per step, and at about 55% accuracy even a pair of answers often
disagrees. (3) At lr 1e-5 for 300 iterations the policy moves so little (KL about 0.017) that the β = 0.04 penalty
(about 0.0007 in the loss) never acts; it would matter for longer or faster training. More SFT moved the policy
twice as far (KL 0.033) and gained far more.

---
## Phase 15 — Error analysis

**Learn:** every response here is a short chain of steps that can be checked exactly, so the error analysis is
automatic and deterministic (`grpo/errors.py`): each of the 16,000 sampled answers per system is compared with the
problem's true computation.

**Run:** `python -m grpo.errors --split test --systems ...` → `results/metrics/errors_test.json`, figure 7

| system | wrong answers (of 16,000) | arithmetic error | misread a number | wrong plan | right steps, wrong final | "lucky" (right answer, wrong steps) | distinct answers per 16 | questions with one answer |
|---|---|---|---|---|---|---|---|---|
| sft_1000 | 7,418 | 96% | 2% | 1% | 1% | 2.1% of correct | 4.6 | 14% |
| grpo_early_s0 | 4,784 | 97% | 1% | 1% | 1% | 1.2% | 3.1 | 34% |
| sft_mid | 1,358 | 98% | 0.4% | 0.4% | 0.1% | 0.1% | 1.6 | 69% |
| sft | 497 | 98% | 1% | 1% | 1% | 0.05% | 1.2 | 82% |
| grpo_s0 | 424 | 97% | 0.5% | 1% | 1% | 0.1% | 1.2 | 84% |

What it shows:
* **Almost every mistake is plain arithmetic** (`8 + 87 = 85`). The model sets up the right steps in the right order
  about 99% of the time; what it lacks is reliable addition and multiplication of 2–3 digit numbers. GRPO reduces the
  *number* of arithmetic errors (7,153 → 4,619) but does not change their *kind*.
* **"Lucky" answers are rare but real:** e.g. `88 + 91 = 189. 189 − 86 = 93` reaches the right answer 93 through two
  wrong steps that cancel. A reward that only checks the final answer pays for this; it falls from 2.1% to 1.2% of
  correct answers after GRPO, so GRPO did not learn to exploit it (no reward hacking).
* **Concentration:** GRPO makes answers less varied (4.6 → 3.1 distinct per 16; questions with a single answer
  14% → 34%); the model commits to its best guess. More SFT does this even more (1.6), while also being more right.
* **No format hacking, no length drift:** invalid answers are under 0.1% everywhere after SFT, and mean length is 38.3
  tokens for every post-trained model.
* Hardest families: multiplication and 3-number chains (sft_1000 44% and 36% sampled; after GRPO 56% and 52%).

---
## Phase 16 — Repository and reproducibility

**Files:** `README.md`, `requirements.txt`, `requirements.lock.txt`, `.gitignore`, `configs/`, `scripts/`,
`tests/`, `results/` (metrics, figures, generations), `project8.html` (tutorial)

**Reproduce everything (CPU, about 6 hours):**
```
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.lock.txt
python scripts/env_report.py
python -m grpo.dataset                                  # data/ (100k train, 500 val, 1,000 test)
python -m grpo.evaluate --ckpt base --name base --split val
python -m grpo.train_sft --config sft.yaml              # about 38 min
bash scripts/lr_pilot.sh                                # about 10 min
bash scripts/train_all.sh                               # 10 GRPO runs + sft_extra, about 2.7 h
bash scripts/eval_all.sh                                # 17 systems on test, about 1 h
python -m grpo.compare --split test
python -m grpo.errors --split test --systems base sft_1000 grpo_early_s0 sft_mid sft grpo_s0 sft_extra
python -m grpo.drift --systems grpo_early_s0 grpo_early_s1 grpo_early_s2 early_beta0 early_nofmt early_g2 early_g4
python -m grpo.drift --ref sft --systems grpo_s0 grpo_s1 grpo_s2 sft_extra --out drift_main.json
python -m grpo.benchmark --systems base sft_1000=sft/step_01000 sft grpo_s0 grpo_early_s0 sft_extra --runs sft grpo_s0
python -m grpo.plots
pytest                                                  # 46 tests
```
Requires Project 1's `checkpoints/small/model_best.pt` (sha256 b4a9986a…611361) and `data/tokenizer.json` next to
this folder (read only). Every random choice has a fixed seed; the data split is a hash of the problem, so it does
not depend on the order of generation. Checkpoints (about 55 MB each) are not in the repository.
