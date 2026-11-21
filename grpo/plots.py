"""Figures (Okabe-Ito colours, colour-blind safe; every series also has its own marker or line style).

  fig1_sft_curve        SFT validation accuracy vs updates; the two GRPO starting points marked
  fig2_grpo_training    GRPO training reward (correct share) and zero-variance share vs iteration, both starts
  fig3_matched_compute  validation sampled accuracy vs training seconds from the step-1,000 start: GRPO vs more SFT
  fig4_test_accuracy    test greedy / sampled / pass@16 for the main systems (95% CI)
  fig5_pass_at_k        test pass@k, k = 1..16
  fig6_ablations        test sampled accuracy, early-start ablations, paired difference to grpo_early (95% CI)
  fig7_errors           what the wrong answers are (share of all answers), per system

  python -m grpo.plots   -> results/figures/*.png
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import grpo  # noqa: E402

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}


def save(fig, name):
    os.makedirs(grpo.FIGURES, exist_ok=True)
    fig.savefig(os.path.join(grpo.FIGURES, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def log(run):
    return list(csv.DictReader(open(os.path.join(grpo.METRICS, "train", run, "log.csv"))))


def ev(name, split="test"):
    p = os.path.join(grpo.METRICS, "eval", split, name + ".json")
    return grpo.load_json(p) if os.path.exists(p) else None


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)


def fig1():
    rows = [r for r in log("sft") if r["val_greedy_acc"]]
    x = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, [100 * float(r["val_greedy_acc"]) for r in rows], color=OI["blue"], marker="o", lw=2, label="greedy")
    ax.plot(x, [100 * float(r["val_sample_acc"]) for r in rows], color=OI["orange"], marker="s", ls="--", lw=2,
            label="sampled (T = 1.0)")
    for s, lab in [(1000, "early start"), (3000, "main start")]:
        ax.axvline(s, color=OI["grey"], ls=":", lw=1)
        ax.text(s, 102, lab, ha="center", fontsize=8, color="#555555")
    ax.set_ylim(0, 108)
    ax.set_xlabel("SFT updates (64 worked solutions each)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("SFT warm-up: the TinyStories model learns the task", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    save(fig, "fig1_sft_curve")


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for runs, color, lab in [(["grpo_early_s0", "grpo_early_s1", "grpo_early_s2"], OI["blue"], "early start"),
                             (["grpo_s0", "grpo_s1", "grpo_s2"], OI["red"], "main start")]:
        for k, run in enumerate(runs):
            if not os.path.exists(os.path.join(grpo.METRICS, "train", run, "log.csv")):
                continue
            rows = [r for r in log(run) if r["reward"]]
            it = np.array([int(r["iter"]) for r in rows])
            for ax, key in zip(axes, ["correct", "zero_var"]):
                y = np.array([float(r[key]) for r in rows])
                sm = np.convolve(y, np.ones(10) / 10, mode="valid")
                ax.plot(it[9:], 100 * sm, color=color, lw=1.6, ls=["-", "--", ":"][k],
                        label=f"{lab}, seed {k}")
    axes[0].set_ylabel("training answers correct (%)")
    axes[1].set_ylabel("groups with no signal (%)")
    axes[0].set_title("Reward rises (10-iteration mean)", fontsize=11)
    axes[1].set_title("Share of all-equal groups (advantage 0)", fontsize=11)
    for ax in axes:
        ax.set_xlabel("GRPO iteration (128 answers each)")
        style(ax)
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    save(fig, "fig2_grpo_training")


def fig3():
    sft = [r for r in log("sft") if r["val_greedy_acc"]]
    t0 = next(float(r["secs"]) for r in sft if int(r["step"]) == 1000)
    xs = [(float(r["secs"]) - t0) / 60 for r in sft if int(r["step"]) >= 1000]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, title in [(axes[0], "val_sample_acc", "sampled (T = 1.0)"), (axes[1], "val_greedy_acc", "greedy")]:
        ax.plot(xs, [100 * float(r[key]) for r in sft if int(r["step"]) >= 1000], color=OI["orange"], marker="s",
                lw=2, label="more SFT (same run, continued)")
        for k in range(3):
            run = f"grpo_early_s{k}"
            if not os.path.exists(os.path.join(grpo.METRICS, "train", run, "log.csv")):
                continue
            rows = [r for r in log(run) if r["val_greedy_acc"]]
            ax.plot([float(r["train_s"]) / 60 for r in rows], [100 * float(r[key]) for r in rows], color=OI["blue"],
                    marker="o", lw=1.8, ls=["-", "--", ":"][k], label=f"GRPO, seed {k}")
        ax.set_xlabel("training minutes after SFT step 1,000 (this CPU)")
        ax.set_title(f"Validation accuracy, {title}", fontsize=11)
        style(ax)
    axes[0].set_ylabel("validation accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig3_matched_compute")


MAIN = [("base", "original LM"), ("sft_1000", "SFT step 1k"), ("grpo_early_s0", "+ GRPO"),
        ("sft_mid", "+ more SFT"), ("sft", "SFT final"), ("grpo_s0", "+ GRPO "), ("sft_extra", "+ more SFT ")]


def _ci(name, metric):
    import grpo.compare as C

    recs = grpo.read_jsonl(grpo.GENERATIONS, "test", name + ".jsonl")
    v = C.per_question(recs, metric)
    rng = np.random.default_rng(0)
    b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
    return 100 * v.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def fig4():
    names = [n for n, _ in MAIN if ev(n)]
    labels = [lab for n, lab in MAIN if ev(n)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.26
    for j, (metric, color, lab, hatch) in enumerate([("greedy", OI["blue"], "greedy", ""),
                                                     ("sample", OI["orange"], "sampled (T = 1.0)", "//"),
                                                     ("pass@16", OI["green"], "pass@16", "..")]):
        vals = [_ci(n, metric) for n in names]
        x = np.arange(len(names)) + (j - 1) * w
        ax.bar(x, [v[0] for v in vals], w - 0.02, color=color, hatch=hatch, edgecolor="white", label=lab)
        ax.errorbar(x, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt="none", ecolor="#333333", capsize=2, lw=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(labels, fontsize=9)
    for s in (0.5, 3.5):
        ax.axvline(s, color=OI["grey"], lw=0.8)
    ax.text(2, 104, "from SFT step 1,000", ha="center", fontsize=9, color="#555555")
    ax.text(5, 104, "from final SFT", ha="center", fontsize=9, color="#555555")
    ax.set_ylim(0, 110)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Held-out accuracy (1,000 test problems, 95% CI)", fontsize=11)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(ax)
    save(fig, "fig4_test_accuracy")


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, sys, title in [(axes[0], [("sft_1000", "SFT step 1k", OI["orange"], "s", "--"),
                                      ("grpo_early_s0", "+ GRPO", OI["blue"], "o", "-"),
                                      ("sft_mid", "+ more SFT", OI["green"], "D", ":"),
                                      ("sft_1000_t07", "SFT step 1k, T = 0.7", OI["purple"], "^", "-.")],
                            "from SFT step 1,000"),
                           (axes[1], [("sft", "SFT final", OI["orange"], "s", "--"),
                                      ("grpo_s0", "+ GRPO", OI["blue"], "o", "-"),
                                      ("sft_extra", "+ more SFT", OI["green"], "D", ":"),
                                      ("sft_t07", "SFT final, T = 0.7", OI["purple"], "^", "-.")],
                            "from final SFT")]:
        for n, lab, c, m, ls in sys:
            e = ev(n)
            if not e:
                continue
            ks = [k for k in (1, 2, 4, 8, 16) if f"pass@{k}" in e]
            ax.plot(ks, [100 * e[f"pass@{k}"] for k in ks], color=c, marker=m, ls=ls, lw=2, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlabel("k (samples per question)")
        ax.set_title(f"pass@k, {title}", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        style(ax)
    axes[0].set_ylabel("test pass@k (%)")
    save(fig, "fig5_pass_at_k")


def fig6():
    p = os.path.join(grpo.METRICS, "test_comparisons.json")
    if not os.path.exists(p):
        return
    cmp_ = {(c["a"], c["b"]): c for c in grpo.load_json(p)}
    rows = [("early_nofmt", "no format bonus"), ("early_g4", "group size 4"), ("early_g2", "group size 2"),
            ("early_beta0", "no KL penalty (β = 0)")]
    rows = [(n, lab) for n, lab in rows if ("grpo_early_s0", n) in cmp_]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (n, lab) in enumerate(rows):
        c = cmp_[("grpo_early_s0", n)]["sample"]
        d, lo, hi = -100 * c["diff"], -100 * c["ci95"][1], -100 * c["ci95"][0]  # ablation minus main
        ax.errorbar([d], [i], xerr=[[d - lo], [hi - d]], fmt="o", color=OI["blue"], capsize=4, ms=7)
        ax.text(hi + 0.3, i, f"{d:+.1f}", va="center", fontsize=9)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows])
    ax.invert_yaxis()
    ax.set_xlabel("test sampled accuracy: ablation − main GRPO (points, 95% CI)")
    ax.set_title("Ablations from the early start (G = 8, β = 0.04, format 0.1)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig6_ablations")


def fig7():
    p = os.path.join(grpo.METRICS, "errors_test.json")
    if not os.path.exists(p):
        return
    e = grpo.load_json(p)
    names = [n for n, _ in MAIN if n in e and n != "base"]
    labels = [lab for n, lab in MAIN if n in e and n != "base"]
    cats = [("arithmetic", OI["red"], ""), ("misread", OI["orange"], "//"), ("wrong_plan", OI["purple"], ".."),
            ("final_mismatch", OI["sky"], "xx"), ("invalid", OI["grey"], "--"), ("other", OI["yellow"], "\\\\")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    bottom = np.zeros(len(names))
    for cat, color, hatch in cats:
        vals = []
        for n in names:
            cnt = e[n]["counts"]
            tot = e[n]["n_responses"]
            v = sum(x for k, x in cnt.items() if k == cat or (cat == "invalid" and k.startswith("invalid")))
            vals.append(100 * v / tot)
        ax.bar(range(len(names)), vals, 0.6, bottom=bottom, color=color, hatch=hatch, edgecolor="white", label=cat)
        bottom += vals
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("wrong answers (% of all sampled answers)")
    ax.set_title("What goes wrong (test, 16 samples per question, automatic step check)", fontsize=11)
    ax.legend(frameon=False, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(ax)
    save(fig, "fig7_errors")


def main():
    fig1()
    fig2()
    fig3()
    for f in (fig4, fig5, fig6, fig7):
        try:
            f()
        except FileNotFoundError as err:
            print("skip", f.__name__, err)
    print("figures ->", grpo.FIGURES)


if __name__ == "__main__":
    main()
