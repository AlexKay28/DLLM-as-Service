#!/usr/bin/env python3
"""Compute AR-30B statistical results for issue #121.

Loads per-event JSONL from the 3-seed BGL run, the HDFS run, and the
DG n=2000 artifact. Computes:
  - Bootstrap 95% CI on AR-30B p50
  - Permutation test (Mann-Whitney U equivalent) AR-30B vs DG
  - Seed-to-seed p50 variance
  - Tail statistics (max/median, CV, p99.9)
  - HDFS vs BGL comparison

Outputs results/anchors/ar30b_stats.json
"""

import json
import statistics
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ANCHORS = REPO / "results" / "anchors"


def load_latencies(path):
    """Load per-event latency_ms values from a JSONL predictions file."""
    lats = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                lats.append(json.loads(line)["latency_ms"])
    return lats


def percentile(values, p):
    """Percentile using numpy (matches numpy.percentile default)."""
    return float(np.percentile(values, p))


def bootstrap_ci(values, stat_fn, n_resample=10000, ci=95.0, seed=42):
    """Bootstrap confidence interval for any statistic."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    stats = []
    for _ in range(n_resample):
        sample = rng.choice(values, size=n, replace=True)
        stats.append(stat_fn(sample))
    lo = percentile(stats, (100 - ci) / 2)
    hi = percentile(stats, 100 - (100 - ci) / 2)
    return round(lo, 1), round(hi, 1)


def permutation_test(a, b, n_perm=100000, seed=42):
    """Two-sided permutation test on difference of medians.

    H0: the two distributions are the same.
    Returns (observed_diff, p_value).
    """
    rng = np.random.RandomState(seed)
    a = np.array(a)
    b = np.array(b)
    obs_diff = abs(np.median(a) - np.median(b))
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = abs(np.median(combined[:na]) - np.median(combined[na:]))
        if perm_diff >= obs_diff:
            count += 1
    p_value = count / n_perm
    return round(obs_diff, 1), round(p_value, 6)


def main():
    # Load AR-30B BGL per-event data (3 seeds)
    seeds = [42, 137, 314]
    bgl_all = []
    bgl_per_seed = {}
    for s in seeds:
        lats = load_latencies(ANCHORS / f"ar30b_L256_s{s}_preds.jsonl")
        bgl_per_seed[s] = lats
        bgl_all.extend(lats)
        print(f"  AR-30B BGL seed {s}: n={len(lats)}, "
              f"p50={percentile(lats, 50):.1f}ms, "
              f"p99={percentile(lats, 99):.1f}ms")

    # Load DG n=2000 data
    dg_lats = load_latencies(ANCHORS / "dg_L256_r1_preds.jsonl")
    print(f"\n  DG n={len(dg_lats)}, p50={percentile(dg_lats, 50):.1f}ms, "
          f"p99={percentile(dg_lats, 99):.1f}ms")

    # Load AR-30B HDFS
    hdfs_lats = load_latencies(ANCHORS / "ar30b_hdfs_L256_s42_preds.jsonl")
    print(f"  AR-30B HDFS n={len(hdfs_lats)}, "
          f"p50={percentile(hdfs_lats, 50):.1f}ms")

    # --- Bootstrap 95% CI on AR-30B p50 ---
    ar30b_p50 = percentile(bgl_all, 50)
    ar30b_ci_lo, ar30b_ci_hi = bootstrap_ci(
        bgl_all, lambda v: percentile(v, 50), n_resample=10000)
    print(f"\n=== AR-30B BGL (3 seeds, n={len(bgl_all)}) ===")
    print(f"  p50 = {ar30b_p50:.1f}ms")
    print(f"  Bootstrap 95% CI: [{ar30b_ci_lo}, {ar30b_ci_hi}] ms")

    # --- Bootstrap 95% CI on DG p50 ---
    dg_p50 = percentile(dg_lats, 50)
    dg_ci_lo, dg_ci_hi = bootstrap_ci(
        dg_lats, lambda v: percentile(v, 50), n_resample=10000)
    print(f"\n=== DG BGL (n={len(dg_lats)}) ===")
    print(f"  p50 = {dg_p50:.1f}ms")
    print(f"  Bootstrap 95% CI: [{dg_ci_lo}, {dg_ci_hi}] ms")

    # --- Permutation test (AR-30B vs DG) ---
    obs_diff, p_val = permutation_test(bgl_all, dg_lats, n_perm=100000)
    print(f"\n=== Permutation test (AR-30B vs DG, difference of medians) ===")
    print(f"  Observed |median diff| = {obs_diff}ms")
    print(f"  p-value = {p_val} (100k permutations)")

    # --- Seed variance ---
    seed_p50s = [percentile(bgl_per_seed[s], 50) for s in seeds]
    seed_mean = round(statistics.fmean(seed_p50s), 2)
    seed_std = round(statistics.stdev(seed_p50s), 2) if len(seed_p50s) > 1 else 0.0
    seed_cv = round(seed_std / seed_mean * 100, 2) if seed_mean > 0 else 0.0
    print(f"\n=== Seed variance (AR-30B BGL, 3 seeds) ===")
    print(f"  Per-seed p50: {[round(p, 1) for p in seed_p50s]}")
    print(f"  Mean p50: {seed_mean:.1f}ms, std: {seed_std:.2f}ms, CV: {seed_cv}%")

    # --- AR-30B tail statistics ---
    ar30b_max = max(bgl_all)
    ar30b_p99 = percentile(bgl_all, 99)
    ar30b_p999 = percentile(bgl_all, 99.9)
    ar30b_max_med = round(ar30b_max / ar30b_p50, 2)
    ar30b_cv = round(
        statistics.stdev(bgl_all) / statistics.fmean(bgl_all) * 100, 2)
    print(f"\n=== AR-30B tail statistics (n={len(bgl_all)}) ===")
    print(f"  p99 = {ar30b_p99:.1f}ms, p99.9 = {ar30b_p999:.1f}ms, "
          f"max = {ar30b_max:.1f}ms")
    print(f"  max/median = {ar30b_max_med}x, CV = {ar30b_cv}%")

    # --- DG tail statistics ---
    dg_max = max(dg_lats)
    dg_p99 = percentile(dg_lats, 99)
    dg_p999 = percentile(dg_lats, 99.9)
    dg_max_med = round(dg_max / dg_p50, 2)
    dg_cv = round(
        statistics.stdev(dg_lats) / statistics.fmean(dg_lats) * 100, 2)
    print(f"\n=== DG tail statistics (n={len(dg_lats)}) ===")
    print(f"  p99 = {dg_p99:.1f}ms, p99.9 = {dg_p999:.1f}ms, "
          f"max = {dg_max:.1f}ms")
    print(f"  max/median = {dg_max_med}x, CV = {dg_cv}%")

    # --- HDFS comparison ---
    hdfs_p50 = percentile(hdfs_lats, 50)
    hdfs_ci_lo, hdfs_ci_hi = bootstrap_ci(
        hdfs_lats, lambda v: percentile(v, 50), n_resample=10000)
    print(f"\n=== AR-30B HDFS (n={len(hdfs_lats)}) ===")
    print(f"  p50 = {hdfs_p50:.1f}ms")
    print(f"  Bootstrap 95% CI: [{hdfs_ci_lo}, {hdfs_ci_hi}] ms")

    # --- Summary JSON ---
    result = {
        "experiment": "issue121_ar30b_stats",
        "ar30b_bgl": {
            "n_events": len(bgl_all),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "p50_ms": round(ar30b_p50, 1),
            "p50_ci_95": [ar30b_ci_lo, ar30b_ci_hi],
            "p99_ms": round(ar30b_p99, 1),
            "p99_9_ms": round(ar30b_p999, 1),
            "max_ms": round(ar30b_max, 1),
            "max_median_ratio": ar30b_max_med,
            "cv_pct": ar30b_cv,
            "seed_p50s": [round(p, 1) for p in seed_p50s],
            "seed_p50_mean": seed_mean,
            "seed_p50_std": seed_std,
            "seed_p50_cv_pct": seed_cv,
        },
        "dg_bgl": {
            "n_events": len(dg_lats),
            "p50_ms": round(dg_p50, 1),
            "p50_ci_95": [dg_ci_lo, dg_ci_hi],
            "p99_ms": round(dg_p99, 1),
            "p99_9_ms": round(dg_p999, 1),
            "max_ms": round(dg_max, 1),
            "max_median_ratio": dg_max_med,
            "cv_pct": dg_cv,
        },
        "permutation_test": {
            "test": "difference of medians, two-sided",
            "n_permutations": 100000,
            "observed_diff_ms": obs_diff,
            "p_value": p_val,
        },
        "ar30b_hdfs": {
            "n_events": len(hdfs_lats),
            "p50_ms": round(hdfs_p50, 1),
            "p50_ci_95": [hdfs_ci_lo, hdfs_ci_hi],
        },
        "timestamp": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    }

    out_path = ANCHORS / "ar30b_stats.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
