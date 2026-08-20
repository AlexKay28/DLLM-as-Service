#!/usr/bin/env python3
"""Generate fig12_distributions: AR-30B vs DG per-event latency box plots.

Shows the full per-event distributions (not just point estimates) for
both arms, with p50, p99, and bootstrap 95% CI markers. Directly
addresses rubric item V11.

Run:  python3 paper/figures/make_distributions_fig.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

ANCHORS = REPO / "results" / "anchors"
OUT = REPO / "paper" / "figures"


def load_latencies(path):
    lats = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                lats.append(json.loads(line)["latency_ms"])
    return lats


def percentile(values, p):
    return float(np.percentile(values, p))


def bootstrap_ci(values, n_resample=10000, seed=42):
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    meds = []
    for _ in range(n_resample):
        sample = rng.choice(values, size=n, replace=True)
        meds.append(np.median(sample))
    lo = percentile(meds, 2.5)
    hi = percentile(meds, 97.5)
    return round(lo, 1), round(hi, 1)


def main():
    S.apply()

    # Load AR-30B (3 seeds pooled) and DG per-event data
    ar30b = []
    for s in [42, 137, 314]:
        ar30b.extend(load_latencies(ANCHORS / f"ar30b_L256_s{s}_preds.jsonl"))
    dg = load_latencies(ANCHORS / "dg_L256_r1_preds.jsonl")

    ar_p50 = percentile(ar30b, 50)
    ar_p99 = percentile(ar30b, 99)
    ar_ci_lo, ar_ci_hi = bootstrap_ci(ar30b)
    dg_p50 = percentile(dg, 50)
    dg_p99 = percentile(dg, 99)
    dg_ci_lo, dg_ci_hi = bootstrap_ci(dg)

    fig, ax = plt.subplots(figsize=(S.FULL, 2.6))

    positions = [1, 2]
    data = [ar30b, dg]
    colors = [S.AR, S.DG]

    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=1.5,
                                                     alpha=0.3, markeredgecolor="none"),
                    medianprops=dict(color=S.INK, linewidth=1.2),
                    whiskerprops=dict(color=S.INK2, linewidth=0.8),
                    capprops=dict(color=S.INK2, linewidth=0.8),
                    boxprops=dict(linewidth=0.8))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)

    # CI brackets at the top
    for i, (lo, hi, p50) in enumerate(
            [(ar_ci_lo, ar_ci_hi, ar_p50),
             (dg_ci_lo, dg_ci_hi, dg_p50)]):
        x = positions[i]
        color = colors[i]
        ax.plot([x, x], [lo, hi], color=color, linewidth=2, solid_capstyle="round")
        ax.plot([x - 0.15, x + 0.15], [lo, lo], color=color, linewidth=1)
        ax.plot([x - 0.15, x + 0.15], [hi, hi], color=color, linewidth=1)

    # p99 markers
    for i, p99 in enumerate([ar_p99, dg_p99]):
        ax.plot(positions[i], p99, marker="_", markersize=8,
                color=colors[i], markeredgewidth=1.5, zorder=5)

    S.despine(ax)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"AR-30B\n(n={len(ar30b)}, 3 seeds)",
                        f"DG-26B\n(n={len(dg)})"])
    ax.set_ylabel("Per-event latency (ms)")
    ax.set_xlim(0.5, 2.5)

    # Annotations
    ax.annotate(f"p50={ar_p50:.0f}\nCI [{ar_ci_lo}, {ar_ci_hi}]",
                xy=(1, ar_p50), xytext=(1.35, ar_p50),
                fontsize=7.5, color=S.AR, va="center")
    ax.annotate(f"p99={ar_p99:.0f}",
                xy=(1, ar_p99), xytext=(1.35, ar_p99),
                fontsize=7.5, color=S.AR, va="center")
    ax.annotate(f"p50={dg_p50:.0f}\nCI [{dg_ci_lo}, {dg_ci_hi}]",
                xy=(2, dg_p50), xytext=(2.35, dg_p50),
                fontsize=7.5, color=S.DG, va="center")
    ax.annotate(f"p99={dg_p99:.0f}",
                xy=(2, dg_p99), xytext=(2.35, dg_p99),
                fontsize=7.5, color=S.DG, va="center")

    ax.set_title("Per-event latency distributions (BGL, L=256, serial batch-1)",
                 loc="left", color=S.INK, fontweight="medium")

    S.save(fig, "fig12_distributions", OUT)


if __name__ == "__main__":
    main()
