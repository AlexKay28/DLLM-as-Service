#!/usr/bin/env python3
"""Build the use-case figures (fig10, fig11) from committed artifacts."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import style as S

S.apply()

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures"


def fig10_chains():
    """Per-hop latency vs chain depth, and tail concentration."""
    with open(RESULTS / "chain_analysis.json") as f:
        d = json.load(f)

    pe1 = d["P_E1_linearity"]
    pe2 = d["P_E2_tail_concentration"]

    depths = pe1["dg"]["depths"]
    dg_p50 = pe1["dg"]["cumul_p50_per_depth"]
    ar_p50 = pe1["ar"]["cumul_p50_per_depth"]

    tail_depths = pe2["depths"]
    p999_ratios = pe2["p999_ratio_per_depth"]
    p50_ratios = pe2["p50_ratio_per_depth"]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.FULL, 2.6),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.35})

    # Panel A: per-hop p50 vs depth
    axA.plot(depths, dg_p50, color=S.DG, lw=1.8, marker=S.MARK["dg"], ms=5,
             mec=S.SURFACE, mew=0.8, zorder=4, label="DG-26B-A4B")
    axA.plot(depths, ar_p50, color=S.AR, lw=1.8, marker=S.MARK["ar"], ms=5,
             mec=S.SURFACE, mew=0.8, zorder=3, label="AR-3B")
    axA.set_xlabel("chain depth")
    axA.set_ylabel("per-hop p50 latency (ms)")
    axA.set_xticks(depths)
    S.title(axA, "A  Per-hop latency is constant across depth")
    S.despine(axA)
    S.label_end(axA, depths[-1], dg_p50[-1], "  DG", S.DG, dx=4, dy=0, size=8)
    S.label_end(axA, depths[-1], ar_p50[-1], "  AR", S.AR, dx=4, dy=0, size=8)

    # Panel B: tail concentration
    axB.plot(tail_depths, p999_ratios, color=S.DG, lw=1.8, marker=S.MARK["dg"],
             ms=5, mec=S.SURFACE, mew=0.8, zorder=4, label="p99.9/p50 ratio")
    axB.axhline(1.93, color=S.AR, lw=1.4, ls="--", zorder=2,
                label="AR mean ratio (1.93×)")
    axB.set_xlabel("chain depth")
    axB.set_ylabel("p99.9 / p50 ratio")
    axB.set_xticks(tail_depths)
    axB.set_ylim(0.8, 2.3)
    S.title(axB, "B  DG tail penalty washes out with depth")
    S.despine(axB)
    axB.legend(loc="lower right", fontsize=7.5)

    S.save(fig, "fig10_usecase_chains", OUT)
    print("  wrote fig10_usecase_chains")


def fig11_wide_schema():
    """Latency vs schema width."""
    with open(RESULTS / "wide_schema_analysis.json") as f:
        d = json.load(f)

    dg = d["P_C1_latency"]["dg"]
    ar = d["P_C1_latency"]["ar"]

    widths = [x["n_fields"] for x in dg]
    dg_lat = [x["p50_ms"] for x in dg]
    ar_lat = [x["p50_ms"] for x in ar]

    fig, ax = plt.subplots(figsize=(S.HALF * 1.8, 2.6))

    ax.plot(widths, dg_lat, color=S.DG, lw=1.8, marker=S.MARK["dg"], ms=5,
            mec=S.SURFACE, mew=0.8, zorder=4, label="DG-26B-A4B")
    ax.plot(widths, ar_lat, color=S.AR, lw=1.8, marker=S.MARK["ar"], ms=5,
            mec=S.SURFACE, mew=0.8, zorder=3, label="AR-3B")

    # Shade the gap
    ax.fill_between(widths, dg_lat, ar_lat, color=S.WASH_DG, alpha=0.5, zorder=1)

    ax.set_xlabel("schema width (fields)")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_xticks(widths)
    S.title(ax, "Latency vs schema width")
    S.despine(ax)
    ax.legend(loc="upper left", fontsize=7.5)

    # Annotate ratios
    for w, dl, al in zip(widths, dg_lat, ar_lat):
        ratio = al / dl
        ax.annotate(f"{ratio:.1f}×", xy=(w, (dl + al) / 2),
                    fontsize=7.0, color=S.MUTED, ha="center")

    S.save(fig, "fig11_usecase_schema", OUT)
    print("  wrote fig11_usecase_schema")


if __name__ == "__main__":
    print("building use-case figures...")
    fig10_chains()
    fig11_wide_schema()
    print("done.")
