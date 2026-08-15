#!/usr/bin/env python3
"""Analyze #44 input-length sweep results.

Fits T(L_in) for both AR and DG models and reports ms/prompt-token.
If T grows with L_in, the cost law is two-variable.

Usage:
    python3 models/experiments/analyze_input_sweep.py
"""

import json
import sys
from pathlib import Path

import statistics

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"


def fit_linear(xs, ys):
    """Simple least-squares y = a + b*x."""
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
    # R²
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, r2


def analyze_sweep(path):
    """Analyze a single input-sweep result file."""
    with open(path) as f:
        data = json.load(f)

    model = data.get("model", "?")
    model_type = data.get("model_type", "?")
    sweep = data.get("sweep", [])

    print(f"\n=== {path.name} ===")
    print(f"Model: {model} ({model_type})")
    if data.get("denoising_steps"):
        print(f"K (denoising steps): {data['denoising_steps']}")

    print(f"\n{'L_in (tok)':>12s} {'p50 (ms)':>10s} {'p99 (ms)':>10s} "
          f"{'F1 (9-field)':>14s} {'ms/prompt_tok':>14s}")

    xs = []
    p50s = []
    f1s = []
    for e in sweep:
        pt = e["prompt_token_count"]
        p50 = e["latency_ms"]["p50"]
        p99 = e["latency_ms"]["p99"]
        f1 = e.get("field_f1_9", e.get("field_f1", 0))
        ms_per_tok = p50 / pt if pt > 0 else 0
        print(f"{pt:12d} {p50:10.1f} {p99:10.1f} {f1:14.4f} {ms_per_tok:14.3f}")
        xs.append(pt)
        p50s.append(p50)
        f1s.append(f1)

    if len(xs) >= 3:
        a, b, r2 = fit_linear(xs, p50s)
        print(f"\n  Linear fit: T = {a:.1f} + {b:.3f} * L_in  (R²={r2:.4f})")
        print(f"  ms per prompt-token: {b:.3f}")
        print(f"  Intercept (fixed cost): {a:.1f} ms")

        if b > 0.15:
            print(f"  → T grows with L_in at {b:.3f} ms/prompt-token — TWO-VARIABLE law")
        elif b < 0.01:
            print(f"  → T is flat in L_in — single-variable law")
        else:
            print(f"  → T has a small L_in slope ({b:.3f} ms/prompt-token) — weak two-variable")

    return {"model": model, "fit_intercept": a if len(xs) >= 3 else 0,
            "fit_slope": b if len(xs) >= 3 else 0, "r2": r2 if len(xs) >= 3 else 0}


def main():
    results = {}
    for name, path in [
        ("AR-3B", RESULTS / "issue44_ar3b_input_sweep.json"),
        ("DG (K=8)", RESULTS / "issue44_dg_input_sweep.json"),
    ]:
        if path.exists():
            results[name] = analyze_sweep(path)
        else:
            print(f"\n{name}: {path} not found (GPU run pending)")

    if len(results) >= 2:
        print("\n=== Comparison ===")
        for name, r in results.items():
            print(f"  {name:12s}: slope={r['fit_slope']:.3f} ms/tok  "
                  f"intercept={r['fit_intercept']:.1f} ms  R²={r['r2']:.4f}")

        ar = results.get("AR-3B", {})
        dg = results.get("DG (K=8)", {})
        if ar and dg:
            if dg["fit_slope"] > ar["fit_slope"] * 1.5:
                print("\n  → DG's input-length slope is steeper than AR's.")
                print("    The flagship application (long stack traces) is")
                print("    where the dLLM advantage evaporates.")
            elif dg["fit_slope"] < 0.01:
                print("\n  → DG's cost is flat in input length — the")
                print("    single-variable law holds for DG.")


if __name__ == "__main__":
    main()
