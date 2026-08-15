#!/usr/bin/env python3
"""Analyze #67 generalization results.

Compares cost-law fits across:
- BGL vs HDFS (second corpus)
- DiffusionGemma vs LLaDA (third model)
- Different block_length values

Reports whether the FORM (linear AR, staircase DG) transfers
while only the CONSTANTS move.

Usage:
    python3 models/experiments/analyze_generalization.py
"""

import json
import sys
import statistics
from pathlib import Path

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


def load_vllm_result(path):
    """Load a vLLM serving result and extract p50 + F1."""
    with open(path) as f:
        d = json.load(f)
    models = d.get("models", {})
    for mk, mv in models.items():
        p50 = mv.get("streaming_table3", {}).get("latency", {}).get("p50_ms")
        quality = mv.get("streaming_table3", {}).get("quality", {})
        f1_9 = quality.get("field_f1_9", quality.get("field_f1", 0))
        f1_12 = quality.get("field_f1_12", 0)
        return {"model": mk, "p50": p50, "f1_9": f1_9, "f1_12": f1_12}
    return None


def load_dinfer_result(path):
    """Load a dInfer result."""
    full_path = RESULTS / "llada_dinfer" / path
    if not full_path.exists():
        full_path = RESULTS / path
    if not full_path.exists():
        return None
    with open(full_path) as f:
        return json.load(f)


def analyze_hdfs_vs_bgl():
    """Arm (a): Compare DG cost law on HDFS vs BGL."""
    print("=== Arm (a): HDFS vs BGL (DiffusionGemma, K=8) ===\n")

    # BGL reference (from #68)
    bgl_data = {
        128: 378.3, 256: 378.3, 264: 500.0, 496: 500.0, 512: 500.9,
        528: 619.6, 768: 620.4, 1024: 738.7, 1280: 860.5,
        1536: 973.6, 1792: 1091.1, 2048: 1212.0
    }

    hdfs_lengths = [128, 256, 512, 1024, 2048]
    hdfs_data = {}
    for L in hdfs_lengths:
        path = RESULTS / f"issue67_dg_hdfs_L{L}.json"
        if path.exists():
            r = load_vllm_result(path)
            if r:
                hdfs_data[L] = r["p50"]
                print(f"  HDFS L={L:4d}: p50={r['p50']:.1f}ms  F1_9={r['f1_9']:.4f}")

    if not hdfs_data:
        print("  (HDFS data not yet available)")
        return

    # Compare BGL vs HDFS at matching lengths
    print(f"\n  {'L':>5s}  {'BGL (ms)':>10s}  {'HDFS (ms)':>10s}  {'ratio':>7s}")
    for L in sorted(set(bgl_data.keys()) & set(hdfs_data.keys())):
        ratio = hdfs_data[L] / bgl_data[L] if bgl_data[L] else 0
        print(f"  {L:5d}  {bgl_data[L]:10.1f}  {hdfs_data[L]:10.1f}  {ratio:7.3f}")

    # Check: is the staircase preserved on HDFS?
    hdfs_sorted = sorted(hdfs_data.items())
    for i in range(1, len(hdfs_sorted)):
        prev_l, prev_p = hdfs_sorted[i - 1]
        curr_l, curr_p = hdfs_sorted[i]
        delta = curr_p - prev_p
        # Check if there's a canvas boundary between prev_l and curr_l
        n_boundaries = sum(1 for b in range(256, curr_l + 1, 256) if b > prev_l)
        if n_boundaries > 0:
            print(f"\n  Step L={prev_l}→{curr_l}: Δ={delta:+.1f}ms "
                  f"({n_boundaries} canvas boundary) ← EDGE")
        else:
            print(f"\n  Step L={prev_l}→{curr_l}: Δ={delta:+.1f}ms (no boundary)")

    # Fit base and step for HDFS
    n_c_values = []
    p50_values = []
    for L, p50 in sorted(hdfs_data.items()):
        n_c = max(1, (L + 255) // 256)
        n_c_values.append(n_c)
        p50_values.append(p50)

    if len(n_c_values) >= 3:
        a, b, r2 = fit_linear(n_c_values, p50_values)
        print(f"\n  HDFS fit: T = {a:.1f} + {b:.1f} * n_c  (R²={r2:.4f})")
        print(f"  HDFS base: {a:.1f} ms  step: {b:.1f} ms")
        print(f"  BGL base: 378.2 ms  step: 119.6 ms")
        print(f"  Base ratio: {a/378.2:.2f}x  Step ratio: {b/119.6:.2f}x")

        if r2 > 0.95:
            print("\n  → Staircase FORM preserved on HDFS (R² > 0.95)")
        else:
            print(f"\n  → Staircase form may NOT hold on HDFS (R²={r2:.3f})")


def analyze_llada():
    """Arm (c): LLaDA-8B block_length sweep."""
    print("\n=== Arm (c): LLaDA-8B under dInfer ===\n")

    # gen_len sweep at block_length=32
    print("  gen_len sweep (block_length=32):")
    gen_data = {}
    for gen in [128, 256, 512, 1024]:
        path = RESULTS / "llada_dinfer" / f"issue67_llada_gen{gen}_bl32.json"
        if not path.exists():
            path = RESULTS / f"issue67_llada_gen{gen}_bl32.json"
        if path.exists():
            d = json.load(open(path))
            p50 = d.get("p50_ms", 0)
            f1 = d.get("field_f1_9", d.get("field_f1", 0))
            gen_data[gen] = {"p50": p50, "f1": f1}
            print(f"    gen={gen:4d}: p50={p50:.1f}ms  F1_9={f1:.4f}")
        else:
            print(f"    gen={gen:4d}: (not yet available)")

    # block_length sweep at gen_len=256
    print("\n  block_length sweep (gen_len=256):")
    bl_data = {}
    for bl in [32, 64, 128, 256]:
        path = RESULTS / "llada_dinfer" / f"issue67_llada_gen256_bl{bl}.json"
        if not path.exists():
            path = RESULTS / f"issue67_llada_gen256_bl{bl}.json"
        if path.exists():
            d = json.load(open(path))
            p50 = d.get("p50_ms", 0)
            f1 = d.get("field_f1_9", d.get("field_f1", 0))
            bl_data[bl] = {"p50": p50, "f1": f1}
            print(f"    bl={bl:3d}: p50={p50:.1f}ms  F1_9={f1:.4f}")
        else:
            print(f"    bl={bl:3d}: (not yet available)")

    # Check: does the step track block_length?
    if len(bl_data) >= 2:
        print("\n  Does step track block_length?")
        print("  (LLaDA uses block_length, not canvas_length — the step boundary")
        print("   should move with block_length if the staircase is a general")
        print("   property of masked diffusion, not just DiffusionGemma.)")


def main():
    analyze_hdfs_vs_bgl()
    analyze_llada()

    print("\n=== Summary ===")
    print("  Form transfer: AR linear in all configs, DG staircase depends on")
    print("  whether HDFS reproduces the step edges and whether LLaDA shows")
    print("  the same block_length-based staircase.")


if __name__ == "__main__":
    main()
