#!/usr/bin/env python3
"""
Issue #46 — fit cost laws with bootstrap confidence intervals.

Loads result JSONs from run_vllm_serving.py (with --save-raw) and fits:
  AR:  T(L) = a + b * L           (OLS, with/without intercept)
  DG:  T(L) = base + step * (ceil(L/256) - 1)   (OLS, fixed/free base)

For each fit, reports:
  - point estimate (slope, intercept/step, sigma)
  - 95% bootstrap CI (10,000 resamples)
  - R²
  - fit with/without L in {32, 64} cells
  - fit with/without free intercept

Also computes quality CIs (bootstrap over per-event F1).

Usage:
    python models/experiments/fit_cost_laws.py results/p2_ar3b_L*.json results/appD_ar3b_L*.json --model ar
    python models/experiments/fit_cost_laws.py results/p2_dg_L*.json results/appD_dg_L*.json --model dg
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_cells(paths):
    """Load result JSONs, extract L, p50, raw latencies, per-event F1."""
    cells = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping", file=sys.stderr)
            continue
        d = json.load(open(p))
        for model_key, model_data in d.get("models", {}).items():
            s = model_data.get("streaming_table3", {})
            lat = s.get("latency", {})
            max_tokens = d.get("max_new_tokens", 0)
            cell = {
                "L": max_tokens,
                "p50": lat.get("p50_ms"),
                "mean": lat.get("mean_ms"),
                "n": lat.get("n_samples", 0),
                "raw_latencies": s.get("raw_latencies_ms"),
                "per_event_f1": s.get("per_event_f1"),
                "macro_f1": s.get("quality", {}).get("field_f1"),
                "source": str(p),
                "model": model_key,
                "max_denoising_steps": model_data.get("max_denoising_steps"),
                "min_tokens": d.get("min_tokens"),
                "vllm_version": model_data.get("vllm_version"),
                "prompt_token_count": model_data.get("prompt_token_count"),
            }
            cells.append(cell)
    cells.sort(key=lambda c: c["L"])
    return cells


def fit_ols(x, y):
    """OLS y = a + b*x. Returns (intercept, slope, sigma, r_squared)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 2:
        return (0, 0, 0, 0)
    # Center x to avoid polyfit conditioning warnings
    x_mean = x.mean()
    b, a_centered = np.polyfit(x - x_mean, y, 1)
    a = float(a_centered - b * x_mean)
    b = float(b)
    pred = a + b * x
    resid = y - pred
    sigma = float(np.sqrt(np.mean(resid ** 2))) if n > 2 else 0.0
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (a, b, sigma, r2)


def fit_ols_no_intercept(x, y):
    """OLS y = b*x (no intercept). Returns (0, slope, sigma, r_squared)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 2:
        return (0, 0, 0, 0)
    b = float(np.sum(x * y) / np.sum(x ** 2))
    pred = b * x
    resid = y - pred
    sigma = float(np.sqrt(np.mean(resid ** 2))) if n > 2 else 0.0
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (0.0, b, sigma, r2)


def bootstrap_ci(samples, n_boot=10000, ci=0.95):
    """Bootstrap CI for the mean of samples."""
    if not samples or len(samples) < 2:
        return (float("nan"), float("nan"))
    arr = np.array(samples, dtype=float)
    rng = np.random.default_rng(42)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return (lo, hi)


def bootstrap_fit_ci(x, y, n_boot=10000, ci=0.95, no_intercept=False):
    """Bootstrap CI for OLS slope and intercept by resampling (x,y) pairs."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return {"slope": (float("nan"), float("nan")),
                "intercept": (float("nan"), float("nan")),
                "sigma": (float("nan"), float("nan"))}
    rng = np.random.default_rng(42)
    slopes = []
    intercepts = []
    sigmas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        xs, ys = x[idx], y[idx]
        if no_intercept:
            a, b, s, _ = fit_ols_no_intercept(xs, ys)
        else:
            a, b, s, _ = fit_ols(xs, ys)
        slopes.append(b)
        intercepts.append(a)
        sigmas.append(s)
    alpha = (1 - ci) / 2
    return {
        "slope": (float(np.percentile(slopes, 100 * alpha)),
                  float(np.percentile(slopes, 100 * (1 - alpha)))),
        "intercept": (float(np.percentile(intercepts, 100 * alpha)),
                      float(np.percentile(intercepts, 100 * (1 - alpha)))),
        "sigma": (float(np.percentile(sigmas, 100 * alpha)),
                  float(np.percentile(sigmas, 100 * (1 - alpha)))),
    }


def fit_ar(cells, label, exclude_small_L=False, no_intercept=False):
    """Fit T_AR(L) = a + b*L."""
    pts = [(c["L"], c["p50"]) for c in cells if c["p50"] is not None]
    if exclude_small_L:
        pts = [(L, t) for L, t in pts if L >= 128]
    if not pts:
        print(f"  {label}: no data")
        return
    L_arr = [p[0] for p in pts]
    T_arr = [p[1] for p in pts]

    if no_intercept:
        a, b, s, r2 = fit_ols_no_intercept(L_arr, T_arr)
    else:
        a, b, s, r2 = fit_ols(L_arr, T_arr)

    ci = bootstrap_fit_ci(L_arr, T_arr, no_intercept=no_intercept)

    print(f"\n  {label}:")
    print(f"    T = {a:.2f} + {b:.4f} * L   (sigma={s:.2f} ms, R²={r2:.4f})")
    print(f"    95% CI: slope [{ci['slope'][0]:.4f}, {ci['slope'][1]:.4f}], "
          f"intercept [{ci['intercept'][0]:.2f}, {ci['intercept'][1]:.2f}], "
          f"sigma [{ci['sigma'][0]:.2f}, {ci['sigma'][1]:.2f}]")
    print(f"    cells: L={L_arr}")

    # Per-cell residuals
    for L_val, T_val in pts:
        pred = (a + b * L_val) if not no_intercept else (b * L_val)
        print(f"      L={L_val:>5d}  T={T_val:.1f}  pred={pred:.1f}  resid={T_val-pred:+.1f}")


def fit_dg(cells, label):
    """Fit T_DG(L) = base + step * (ceil(L/256) - 1)."""
    pts = [(c["L"], c["p50"]) for c in cells if c["p50"] is not None]
    if not pts:
        print(f"  {label}: no data")
        return

    L_arr = np.array([p[0] for p in pts], dtype=float)
    T_arr = np.array([p[1] for p in pts], dtype=float)
    n_c = np.maximum(1, np.ceil(L_arr / 256)).astype(int)

    # Base: mean of single-canvas cells
    single_mask = n_c == 1
    base = float(np.mean(T_arr[single_mask])) if np.any(single_mask) else float("nan")

    # Per-cell implied steps
    multi_mask = n_c > 1
    per_cell = {}
    if np.any(multi_mask):
        for i in np.where(multi_mask)[0]:
            k = int(n_c[i])
            s = (T_arr[i] - base) / (k - 1)
            per_cell[k] = float(s)

    # Unweighted mean
    step_mean = float(np.mean(list(per_cell.values()))) if per_cell else float("nan")

    # Least-squares (fixed base)
    if np.any(multi_mask):
        numer = np.sum((T_arr[multi_mask] - base) * (n_c[multi_mask] - 1))
        denom = np.sum((n_c[multi_mask] - 1) ** 2)
        step_ls = float(numer / denom) if denom > 0 else float("nan")
    else:
        step_ls = float("nan")

    # Bootstrap CI for step (resample multi-canvas cells)
    step_ci = (float("nan"), float("nan"))
    if np.any(multi_mask) and np.sum(multi_mask) >= 2:
        rng = np.random.default_rng(42)
        steps_boot = []
        multi_idx = np.where(multi_mask)[0]
        for _ in range(10000):
            idx = rng.choice(multi_idx, size=len(multi_idx), replace=True)
            x = (n_c[idx] - 1).astype(float)
            y = T_arr[idx] - base
            s = float(np.sum(x * y) / np.sum(x ** 2)) if np.sum(x ** 2) > 0 else 0
            steps_boot.append(s)
        step_ci = (float(np.percentile(steps_boot, 2.5)),
                   float(np.percentile(steps_boot, 97.5)))

    print(f"\n  {label}:")
    print(f"    base = {base:.2f} ms (mean of {int(np.sum(single_mask))} single-canvas cells)")
    print(f"    per-cell implied steps: {', '.join(f'n_c={k}: {v:.1f}' for k, v in sorted(per_cell.items()))}")
    print(f"    step (unweighted mean) = {step_mean:.2f} ms")
    print(f"    step (least-squares)   = {step_ls:.2f} ms  "
          f"[95% CI: {step_ci[0]:.1f}, {step_ci[1]:.1f}]")
    if per_cell:
        spread = max(per_cell.values()) - min(per_cell.values())
        print(f"    step spread: {spread:.1f} ms ({100*spread/min(per_cell.values()):.1f}%)")

    # Asymptotic ratio for each step
    ar_slope = 6.29
    for name, step in [("mean", step_mean), ("LS", step_ls)]:
        if not math.isnan(step):
            ratio = ar_slope / (step / 256)
            print(f"    asymptotic ratio ({name}): {ratio:.1f}x")


def quality_cis(cells):
    """Bootstrap CIs for quality (per-event F1)."""
    print(f"\n  Quality CIs (bootstrap over per-event F1):")
    for c in cells:
        f1s = c.get("per_event_f1")
        if not f1s:
            continue
        lo, hi = bootstrap_ci(f1s)
        print(f"    L={c['L']:>5d}  F1={c['macro_f1']:.4f}  "
              f"[95% CI: {lo:.4f}, {hi:.4f}]  n={len(f1s)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="result JSON paths")
    ap.add_argument("--model", choices=["ar", "dg", "both"], default="both")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    cells = load_cells(args.paths)
    if not cells:
        print("No cells found. Run with --save-raw first.")
        sys.exit(1)

    print(f"Loaded {len(cells)} cells:")
    for c in cells:
        has_raw = "yes" if c.get("raw_latencies") else "NO"
        print(f"  L={c['L']:>5d}  p50={c['p50']:.1f}ms  F1={c.get('macro_f1','?')}  "
              f"raw={has_raw}  src={Path(c['source']).name}")

    has_raw = all(c.get("raw_latencies") for c in cells)
    if not has_raw:
        print("\n  WARNING: not all cells have raw latencies. "
              "Re-run with --save-raw for bootstrap CIs on latency.")

    if args.model in ("ar", "both"):
        print("\n" + "=" * 60)
        print("AR cost law: T(L) = a + b * L")
        print("=" * 60)
        fit_ar(cells, "all cells, with intercept")
        fit_ar(cells, "all cells, no intercept", no_intercept=True)
        fit_ar(cells, "L>=128 only, with intercept", exclude_small_L=True)

    if args.model in ("dg", "both"):
        print("\n" + "=" * 60)
        print("DG cost law: T(L) = base + step * (ceil(L/256) - 1)")
        print("=" * 60)
        fit_dg(cells, "all cells")

    print("\n" + "=" * 60)
    print("Quality CIs")
    print("=" * 60)
    quality_cis(cells)


if __name__ == "__main__":
    main()
