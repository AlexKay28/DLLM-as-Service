#!/usr/bin/env python3
"""Analyze #44 DG K-sweep (base decomposition) and #68 canvas boundary sweep.

Decomposes DG's base into prefill / denoise / overhead:
  T_DG = prefill(L_in) + K * step_cost(canvas) + fixed

Usage:
    python3 models/experiments/analyze_ksweep.py
"""

import json
import sys
from pathlib import Path

import statistics

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"


def fit_linear(xs, ys):
    n = len(xs)
    if n < 2:
        return 0, 0, 0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0, 0, 0
    b = (n * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - b * sum_x) / n
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, r2


def analyze_ksweep():
    """Analyze DG K-sweep: T = base + K * step_cost."""
    print("=== #44 Arm B: DG K-sweep (base decomposition) ===\n")

    ks = []
    p50s = []
    f1s = []

    for k in [2, 4, 8, 16, 32, 64]:
        path = RESULTS / f"issue44_dg_ksweep_K{k}.json"
        if not path.exists():
            print(f"  K={k:2d}: {path.name} not found (pending)")
            continue
        with open(path) as f:
            data = json.load(f)
        mk = list(data.get("models", {}).keys())[0]
        mv = data["models"][mk]
        p50 = mv["streaming_table3"]["latency"]["p50_ms"]
        f1 = mv["streaming_table3"]["quality"].get("field_f1_9",
              mv["streaming_table3"]["quality"].get("field_f1", 0))
        ks.append(k)
        p50s.append(p50)
        f1s.append(f1)
        print(f"  K={k:2d}: p50={p50:.1f} ms  F1={f1:.4f}")

    if len(ks) >= 3:
        a, b, r2 = fit_linear(ks, p50s)
        print(f"\n  Fit: T = {a:.1f} + {b:.2f} * K  (R²={r2:.4f})")
        print(f"  Base (prefill + fixed): {a:.1f} ms")
        print(f"  Per-step cost: {b:.2f} ms")
        print(f"  At K=8: denoising = {8*b:.1f} ms = {100*8*b/a:.1f}% of base")
        print(f"  At K=8: prefill+fixed = {a:.1f} ms = {100*a/(a+8*b):.1f}% of total")

        # F1 analysis
        print(f"\n  F1 range: {min(f1s):.4f} - {max(f1s):.4f} "
              f"(spread: {max(f1s)-min(f1s):.4f})")
        if max(f1s) - min(f1s) < 0.01:
            print("  → F1 is flat across K — step budget is nearly free quality-wise")
        else:
            print(f"  → F1 varies by {max(f1s)-min(f1s):.4f} — step budget matters")

        return {"base": a, "step_cost": b, "r2": r2}
    return None


def analyze_canvas_boundaries():
    """Analyze #68: canvas step edges."""
    print("\n=== #68: Canvas boundary sweep ===\n")

    lengths = [240, 256, 264, 496, 512, 528, 768, 1024, 1280, 1536, 1792, 2048]
    data_points = []

    for L in lengths:
        path = RESULTS / f"issue68_dg_L{L}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        mk = list(data.get("models", {}).keys())[0]
        mv = data["models"][mk]
        p50 = mv["streaming_table3"]["latency"]["p50_ms"]
        f1 = mv["streaming_table3"]["quality"].get("field_f1_9",
              mv["streaming_table3"]["quality"].get("field_f1", 0))
        n_c = (L + 255) // 256  # number of canvases
        data_points.append({"L": L, "p50": p50, "f1": f1, "n_c": n_c})
        print(f"  L={L:4d} (n_c={n_c}): p50={p50:.1f} ms  F1={f1:.4f}")

    if len(data_points) < 3:
        print("  (insufficient data — GPU run pending)")
        return None

    # Check for step edges: does p50 jump at L=256, 512, 768, ...?
    print("\n  Step-edge analysis:")
    for i in range(1, len(data_points)):
        prev = data_points[i - 1]
        curr = data_points[i]
        delta = curr["p50"] - prev["p50"]
        same_canvas = curr["n_c"] == prev["n_c"]
        marker = "  " if same_canvas else "← EDGE"
        print(f"    L={prev['L']:4d}→{curr['L']:4d}: Δp50={delta:+7.1f} ms "
              f"(n_c {prev['n_c']}→{curr['n_c']}) {marker}")

    # Per-canvas step cost
    print("\n  Per-canvas step cost (by n_c):")
    by_nc = {}
    for d in data_points:
        nc = d["n_c"]
        if nc not in by_nc:
            by_nc[nc] = []
        by_nc[nc].append(d["p50"])

    for nc in sorted(by_nc.keys()):
        vals = by_nc[nc]
        med = statistics.median(vals)
        print(f"    n_c={nc}: median p50={med:.1f} ms (n={len(vals)} cells)")

    # Compute step costs between consecutive n_c
    print("\n  Step cost between consecutive canvas counts:")
    sorted_nc = sorted(by_nc.keys())
    for i in range(1, len(sorted_nc)):
        nc_prev = sorted_nc[i - 1]
        nc_curr = sorted_nc[i]
        med_prev = statistics.median(by_nc[nc_prev])
        med_curr = statistics.median(by_nc[nc_curr])
        step = med_curr - med_prev
        print(f"    n_c {nc_prev}→{nc_curr}: step = {step:.1f} ms")

    return data_points


def main():
    ksweep = analyze_ksweep()
    canvas = analyze_canvas_boundaries()

    if ksweep:
        print(f"\n=== Summary ===")
        print(f"  DG base decomposition:")
        print(f"    Prefill + fixed: {ksweep['base']:.1f} ms ({100*ksweep['base']/(ksweep['base']+8*ksweep['step_cost']):.1f}% at K=8)")
        print(f"    Denoising (K=8): {8*ksweep['step_cost']:.1f} ms ({100*8*ksweep['step_cost']/(ksweep['base']+8*ksweep['step_cost']):.1f}% at K=8)")
        print(f"    Per-step cost: {ksweep['step_cost']:.2f} ms")


if __name__ == "__main__":
    main()
