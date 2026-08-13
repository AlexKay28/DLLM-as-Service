#!/usr/bin/env python3
"""
Issue #14: plots + summary stats for P1 (rate-vs-lag) and P2 (length scaling).

Reads the JSON artifacts produced by rate_lag_curve.py and the
run_vllm_serving.py P2 sweep; writes PNG charts to results/ and prints the
crossover/slope summary used in the report. Runs in .venv-dg
(matplotlib dependency — experiment venv only, not the stdlib repo path).

Usage:
    .venv-dg/bin/python models/experiments/plot_issue14.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _REPO_ROOT / "results"

DG_LABEL = "DiffusionGemma-26B (dLLM)"
AR_LABEL = "AR-3B (Qwen2.5)"


def load_rate_lag(path):
    d = json.load(open(path))
    rates, p50, p99, completed = [], [], [], []
    for r in d["rates"]:
        rates.append(r["offered_rate_ev_s"])
        p50.append(r["lag_ms"].get("p50_ms"))
        p99.append(r["lag_ms"].get("p99_ms"))
        completed.append(r["completion_ratio"])
    return np.array(rates), np.array(p50, dtype=float), \
        np.array(p99, dtype=float), np.array(completed, dtype=float)


def load_p2():
    rows = {}
    for L in (32, 64, 128, 256, 512):
        for tag, pat in (("dg", f"p2_dg_L{L}.json"), ("ar", f"p2_ar3b_L{L}.json")):
            d = json.load(open(RESULTS / pat))
            lat = list(d["models"].values())[0]["streaming_table3"]["latency"]
            rows[(tag, L)] = (lat["p50_ms"], lat["p99_ms"])
    return rows


def plot_rate_lag():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f, label, color in (("rate_lag_dg_seq1.json", DG_LABEL, "tab:orange"),
                            ("rate_lag_ar3b.json", AR_LABEL, "tab:blue")):
        rates, p50, p99, _ = load_rate_lag(RESULTS / f)
        ax.plot(rates, p50, "o-", color=color, label=f"{label} p50")
        ax.plot(rates, p99, "s--", color=color, alpha=0.55, label=f"{label} p99")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offered stream rate (events/s, Poisson)")
    ax.set_ylabel("lag: arrival→finish (ms, log)")
    ax.set_title("Issue #14 P1: lag vs offered rate (A100, vLLM 0.25.1, BGL)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "issue14_rate_lag.png", dpi=150)
    print(f"wrote {RESULTS/'issue14_rate_lag.png'}")


def plot_length_scaling():
    rows = load_p2()
    Ls = np.array([32, 64, 128, 256, 512])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, label, color in (("dg", DG_LABEL, "tab:orange"),
                              ("ar", AR_LABEL, "tab:blue")):
        p50 = [rows[(tag, L)][0] for L in Ls]
        p99 = [rows[(tag, L)][1] for L in Ls]
        ax.plot(Ls, p50, "o-", color=color, label=f"{label} p50")
        ax.plot(Ls, p99, "s--", color=color, alpha=0.55, label=f"{label} p99")
    # AR slope fit on p50
    ar_p50 = np.array([rows[("ar", L)][0] for L in Ls])
    b, a = np.polyfit(Ls, ar_p50, 1)
    ax.plot(Ls, a + b * Ls, ":", color="tab:blue", alpha=0.6,
            label=f"AR fit: {a:.0f} + {b:.1f} ms/token")
    xover = None
    dg_p50 = np.array([rows[("dg", L)][0] for L in Ls])
    diff = ar_p50 - dg_p50
    for i in range(len(Ls) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            # linear interpolation of the crossing on the AR fit
            xover = (dg_p50[i] - a) / b
    if xover:
        ax.axvline(xover, color="gray", ls="--", alpha=0.6)
        ax.annotate(f"crossover ≈ {xover:.0f} tok", (xover, ax.get_ylim()[1]*0.5),
                    rotation=90, va="center")
    ax.set_xlabel("forced output length L (tokens)")
    ax.set_ylabel("m=1 latency p50/p99 (ms)")
    ax.set_title("Issue #14 P2: latency vs output length (A100, vLLM 0.25.1, BGL)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "issue14_length_scaling.png", dpi=150)
    print(f"wrote {RESULTS/'issue14_length_scaling.png'}")
    print(f"AR slope: {b:.2f} ms/token (intercept {a:.0f} ms); "
          f"crossover ≈ {xover:.0f} tokens" if xover else "no crossover in range")
    for L in Ls:
        d_, a_ = rows[("dg", L)][0], rows[("ar", L)][0]
        print(f"  L={L:4d}: DG {d_:7.1f} ms, AR {a_:7.1f} ms, ratio {a_/d_:.2f}x")


def crossover_rate():
    """Lowest offered rate where AR's p50 lag beats DG's (both interpolated)."""
    r_dg, p50_dg, _, _ = load_rate_lag(RESULTS / "rate_lag_dg_seq1.json")
    r_ar, p50_ar, _, _ = load_rate_lag(RESULTS / "rate_lag_ar3b.json")
    grid = np.geomspace(max(r_dg.min(), r_ar.min()),
                        min(r_dg.max(), r_ar.max()), 200)
    dg_i = np.interp(np.log(grid), np.log(r_dg), p50_dg)
    ar_i = np.interp(np.log(grid), np.log(r_ar), p50_ar)
    win = dg_i < ar_i
    if win.all():
        return "DG below AR across the whole shared range (no crossover)"
    if (~win).all():
        return "AR below DG across the whole shared range (no crossover)"
    idx = int(np.argmin(win))
    return (f"p50-lag crossover ≈ {grid[idx]:.2f} ev/s "
            f"(DG lower lag below this rate, AR lower above)")


if __name__ == "__main__":
    plot_rate_lag()
    plot_length_scaling()
    print(crossover_rate())
