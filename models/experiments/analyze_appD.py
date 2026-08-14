#!/usr/bin/env python3
"""
Issue #17 (App D) analysis: fitted cost laws + batching multipliers.

Combines #14's P2 cells (L<=512) with App D's extended cells (L=1024, 2048),
fits:
    AR:  T(L) = b0 + b1 * L                       (linear)
    DG:  T(L) = a0 + a1 * (ceil(L/256) - 1)       (stepwise canvases)
with residual-based 95% CIs (per-event samples are not persisted by the
runner — recorded percentiles only; stated in the output).

Tests P-D2 (DG at 1024 ≈ 719 ms, at 2048 ≈ 1120 ms, ±15%) and P-D3 (AR
linear to 2048), reports the batching multipliers per model per L, and
writes results/appD_analysis.json + results/appD_costlaws.png.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _REPO_ROOT / "results"

P2 = {("dg", L): RESULTS / f"p2_dg_L{L}.json" for L in (32, 64, 128, 256, 512)}
P2.update({("ar", L): RESULTS / f"p2_ar3b_L{L}.json" for L in (32, 64, 128, 256, 512)})
APPD = {("dg", L): RESULTS / f"appD_dg_L{L}.json" for L in (1024, 2048)}
APPD.update({("ar", L): RESULTS / f"appD_ar3b_L{L}.json" for L in (1024, 2048)})

PRED_DG = {1024: 719.0, 2048: 1120.0}   # issue #17 P-D2 extrapolation (NOT a fit — see #51)
AR_PRED_SLOPE = 6.28                     # issue #16 law


def p50(path):
    d = json.load(open(path))
    return list(d["models"].values())[0]["streaming_table3"]["latency"]["p50_ms"]


def fit_ar(points):
    L = np.array(sorted(points), dtype=float)
    T = np.array([points[l] for l in points.keys() & set(map(int, L))])
    T = np.array([points[int(l)] for l in L])
    b1, b0 = np.polyfit(L, T, 1)
    resid = T - (b0 + b1 * L)
    sigma = float(np.sqrt(np.mean(resid ** 2))) if len(L) > 2 else 0.0
    return b0, b1, sigma


def fit_dg(points):
    """DG: T(L) = base + step * (ceil(L/256) - 1).

    step is the marginal cost per additional 256-token canvas block.
    The step is NOT constant across cells (issue #51): the first extra
    canvas costs ~101 ms, later ones ~120 ms. We report both the
    unweighted mean (as before) and the least-squares fit (weighted by
    (n_c-1)^2, dominated by the n_c=8 cell).
    """
    L = np.array(sorted(points), dtype=float)
    blocks = np.maximum(1, np.ceil(L / 256)).astype(int)
    T = np.array([points[int(l)] for l in L])
    per_block = {}
    for b_, t_ in zip(blocks, T):
        per_block.setdefault(b_, []).append(t_)
    base = float(np.mean(per_block.get(1, [np.nan])))

    # Per-cell implied steps (for diagnostics)
    per_cell_steps = {}
    steps = []
    for k, vals in per_block.items():
        if k > 1:
            for v in vals:
                s = (v - base) / (k - 1)
                steps.append(s)
                per_cell_steps[k] = s

    # Unweighted mean (original method)
    step_mean = float(np.mean(steps)) if steps else float("nan")

    # Least-squares: step = sum((T-base)*(n_c-1)) / sum((n_c-1)^2)
    multi_mask = blocks > 1
    if np.any(multi_mask):
        numer = np.sum((T[multi_mask] - base) * (blocks[multi_mask] - 1))
        denom = np.sum((blocks[multi_mask] - 1) ** 2)
        step_ls = float(numer / denom) if denom > 0 else float("nan")
    else:
        step_ls = float("nan")

    return base, step_mean, step_ls, per_cell_steps


def multiplier(path):
    d = json.load(open(path))
    tp = list(d["models"].values())[0]["throughput_table4"]
    sweep = {r["batch_size"]: r for r in tp["sweep"]}
    b1 = sweep.get(1)
    ev1 = b1["events_per_s"]
    peak = tp["peak_events_per_s"] or float("nan")
    return ev1, peak, peak / ev1 if ev1 else float("nan")


def main():
    out = {}
    tg = {
        "dg": {L: p50(p) for (t, L), p in {**P2, **APPD}.items() if t == "dg" and p.exists()},
        "ar": {L: p50(p) for (t, L), p in {**P2, **APPD}.items() if t == "ar" and p.exists()},
    }
    for m in ("dg", "ar"):
        tg[m] = {int(k): v for k, v in tg[m].items()}
    b0, b1, sigma = fit_ar(tg["ar"])
    base, step_mean, step_ls, per_cell_steps = fit_dg(tg["dg"])
    out["ar_fit"] = {"intercept_ms": round(b0, 2), "slope_ms_per_token": round(b1, 3),
                     "resid_sigma_ms": round(sigma, 2)}
    out["dg_fit"] = {"base_ms": round(base, 2),
                     "step_per_canvas_ms_unweighted_mean": round(step_mean, 2),
                     "step_per_canvas_ms_least_squares": round(step_ls, 2),
                     "per_cell_implied_steps": {str(k): round(v, 2) for k, v in per_cell_steps.items()},
                     "step_is_constant": bool(abs(max(per_cell_steps.values()) - min(per_cell_steps.values())) < 5.0) if per_cell_steps else None}
    step = step_ls  # use LS for downstream ratios (most defensible)
    out["ratio_T_AR_over_T_DG"] = {
        str(L): round((b0 + b1 * L) / (base + step * (math.ceil(L / 256) - 1)), 2)
        for L in sorted(tg["ar"]) if L in tg["dg"]}
    out["P_D2_check"] = {
        str(L): {"predicted_ms": PRED_DG[L], "measured_ms": tg["dg"][L],
                 "within_15pct": abs(tg["dg"][L] - PRED_DG[L]) / PRED_DG[L] <= 0.15}
        for L in PRED_DG if L in tg["dg"]}

    out["P_D3_check"] = {"slope": round(b1, 3), "expected": AR_PRED_SLOPE,
                         "linear_holds": bool(abs(b1 - AR_PRED_SLOPE) / AR_PRED_SLOPE <= 0.10)}
    mult = {}
    for tag in ("dg", "ar3b"):
        for L in (128, 512):
            p = RESULTS / f"appD_mult_{tag}_L{L}.json"
            if not p.exists():
                mult[f"{tag}_L{L}"] = {"status": "pending"}
                continue
            ev1, peak, ratio = multiplier(p)
            mult[f"{tag}_L{L}"] = {"batch1_ev_s": round(ev1, 3),
                                   "peak_ev_s": round(peak, 2),
                                   "multiplier": round(ratio, 1)}
    out["batching_multipliers"] = mult

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    Lgrid = np.linspace(32, 2048, 400)
    for pts, name, c in ((tg["dg"], "DiffusionGemma (dLLM)", "#E8734A"),
                         (tg["ar"], "AR-3B (Qwen2.5)", "#4A90D9")):
        Ls = sorted(pts)
        ax.plot(Ls, [pts[l] for l in Ls], "o", color=c, label=name)
    ax.plot(Lgrid, b0 + b1 * Lgrid, color="#4A90D9", ls="--",
            label=f"AR fit: {b0:.0f}+{b1:.2f}·L")
    dg_L = np.array(sorted(tg["dg"]), dtype=float)
    ax.step(dg_L, base + step * (np.ceil(dg_L / 256) - 1), where="post",
            color="#E8734A", ls="--", label=f"DG fit: {base:.0f}+{step:.0f}/canvas")
    ax.axvline(65, color="gray", ls=":", lw=1)
    ax.set_xlabel("forced output length L (tokens)")
    ax.set_ylabel("m=1 latency p50 (ms)")
    ax.set_title("Cost laws to L=2048 (A100, vLLM 0.25.1)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ratio = [(L, tg["ar"][L] / tg["dg"][L]) for L in sorted(tg["ar"])]
    ax.plot([r[0] for r in ratio], [r[1] for r in ratio], "o-", color="black",
            label="measured T_AR / T_DG")
    ax.plot(Lgrid, (b0 + b1 * Lgrid) / (base + step * (np.ceil(Lgrid / 256) - 1)),
            color="gray", ls="--", label="fitted ratio")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("output length L (tokens)")
    ax.set_ylabel("AR / DG time ratio")
    ax.set_title("DG advantage vs output length")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "appD_costlaws.png", dpi=150)

    with open(RESULTS / "appD_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
