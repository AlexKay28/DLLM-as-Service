#!/usr/bin/env python3
"""
Issue #21 (App E) analysis: CLT tail-concentration test + mixed-hop simulation.

Reads the GPU chain experiment output (per-hop marginals + per-depth
cumulative latencies) and tests:

  P-E1: cumulative latency linear in D, slope = single-hop p50
  P-E2: cumulative p99.9 ratio migrates from 1.27x (D=1) toward 1.9x
        (mean ratio) as D grows (CLT)
  P-E3: concurrency = 1 (by construction)
  P-E4: per-hop quality unchanged

Then simulates mixed-hop chains (R1): decision hops at L~30 (below
crossover, AR-favoured) interleaved with extraction hops at L~150
(above crossover, DG-favoured), using marginals from #14 P2 cells,
to test whether DG's advantage survives heterogeneous hop lengths.

Outputs results/chain_analysis.json + results/chain_clt.{png,html}.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _REPO_ROOT / "results"


def load_chain(path):
    return json.load(open(path))


def load_p2_marginals():
    """Per-hop latency marginals from #14 P2 at various L."""
    marg = {}
    for tag in ("dg", "ar3b"):
        for L in (32, 64, 128, 256, 512):
            p = RESULTS / f"p2_{tag}_L{L}.json"
            if p.exists():
                d = json.load(open(p))
                s = list(d["models"].values())[0]["streaming_table3"]
                lat = s["latency"]
                marg[f"{tag}_L{L}"] = {
                    "p50": lat["p50_ms"], "p99": lat["p99_ms"],
                    "p99.9": lat["p99_9_ms"],
                    "mean": lat.get("mean_ms", lat["p50_ms"]),
                    "min": lat.get("min_ms", lat["p50_ms"]),
                    "max": lat.get("max_ms", lat["p99_9_ms"]),
                }
    return marg


def fit_lognormal(p50, p99):
    """Fit a lognormal from p50 and p99 (z=2.326 for p99)."""
    mu = math.log(p50)
    z99 = 2.326
    sigma = (math.log(p99) - mu) / z99 if p99 > 0 else 0.3
    return mu, max(sigma, 0.05)


def sample_lognormal(mu, sigma, n, rng):
    return np.exp(rng.normal(mu, sigma, n))


def clt_cumulative_p999(mean, sigma, D, z999=3.09):
    """Analytical CLT: p99.9 of sum of D iid = D*mean + z999*sqrt(D)*sigma."""
    return D * mean + z999 * math.sqrt(D) * sigma


def main():
    dg_chain = load_chain(RESULTS / "chain_dg.json")
    ar_chain = load_chain(RESULTS / "chain_ar3b.json")

    out = {"P_E1_linearity": {}, "P_E2_tail_concentration": {}, "P_E3": {},
           "P_E4_quality": {}, "mixed_hop_simulation": {}}

    # P-E1: linearity — cumulative p50 / D should be constant = single-hop p50
    for label, chain in [("dg", dg_chain), ("ar", ar_chain)]:
        depths = sorted(int(d) for d in chain["per_depth"])
        slopes = []
        for D in depths:
            cum_p50 = chain["per_depth"][str(D)]["cumulative_latency"]["p50_ms"]
            slopes.append(cum_p50 / D)
        mean_slope = float(np.mean(slopes)) if slopes else 0.0
        cv = float(np.std(slopes) / mean_slope) if mean_slope else None
        out["P_E1_linearity"][label] = {
            "depths": depths,
            "cumul_p50_per_depth": slopes,
            "slope_cv": cv,
            "linear": cv is not None and cv < 0.05,
        }

    # P-E2: tail concentration — cumulative p99.9 ratio should migrate
    # from single-hop 1.27x toward mean ratio 1.9x
    depths = sorted(int(d) for d in dg_chain["per_depth"])
    ratios_p999 = []
    ratios_p50 = []
    for D in depths:
        dg_p999 = dg_chain["per_depth"][str(D)]["cumulative_latency"]["p99_9_ms"]
        ar_p999 = ar_chain["per_depth"][str(D)]["cumulative_latency"]["p99_9_ms"]
        dg_p50 = dg_chain["per_depth"][str(D)]["cumulative_latency"]["p50_ms"]
        ar_p50 = ar_chain["per_depth"][str(D)]["cumulative_latency"]["p50_ms"]
        ratios_p999.append(ar_p999 / dg_p999 if dg_p999 > 0 else None)
        ratios_p50.append(ar_p50 / dg_p50 if dg_p50 > 0 else None)
    out["P_E2_tail_concentration"] = {
        "depths": depths,
        "p999_ratio_per_depth": ratios_p999,
        "p50_ratio_per_depth": ratios_p50,
        "single_hop_p999_ratio": ratios_p999[0] if ratios_p999 else None,
        "mean_ratio_p50": ratios_p50[0] if ratios_p50 else None,
        "converges_to_mean": ratios_p999 and ratios_p999[-1] and ratios_p50 and
        abs(ratios_p999[-1] - ratios_p50[0]) < abs(ratios_p999[0] - ratios_p50[0]),
    }

    # P-E3: concurrency = 1 by construction (m=1 sequential calls)
    out["P_E3"] = {"concurrency": 1, "by_construction": True,
                   "note": "Each hop is a single llm.generate([prompt]) call, "
                           "no concurrent requests in flight"}

    # P-E4: quality unchanged across depths
    for label, chain in [("dg", dg_chain), ("ar", ar_chain)]:
        f1s = []
        for D in sorted(int(d) for d in chain["per_depth"]):
            f1s.append(chain["per_depth"][str(D)]["quality"]["field_f1"])
        out["P_E4_quality"][label] = {
            "f1_per_depth": f1s,
            "unchanged": (max(f1s) - min(f1s)) < 0.02,
        }

    # Analytical CLT prediction
    dg_hop = dg_chain["per_depth"]["1"]["per_hop_latency"]
    ar_hop = ar_chain["per_depth"]["1"]["per_hop_latency"]
    out["analytical_CLT"] = {
        "dg": {"mean_ms": dg_hop["mean_ms"], "sigma_ms": None,
               "cumul_p999_predicted": []},
        "ar": {"mean_ms": ar_hop["mean_ms"], "sigma_ms": None,
               "cumul_p999_predicted": []},
    }
    for label, hop, chain in [("dg", dg_hop, dg_chain), ("ar", ar_hop, ar_chain)]:
        mean = hop["mean_ms"]
        # estimate sigma from p99.9 - p50 spread
        sigma = (hop["p99_9_ms"] - hop["p50_ms"]) / 3.09
        out["analytical_CLT"][label]["sigma_ms"] = round(sigma, 1)
        for D in depths:
            pred = clt_cumulative_p999(mean, sigma, D)
            out["analytical_CLT"][label]["cumul_p999_predicted"].append(
                round(pred, 0))

    # Mixed-hop simulation (R1): interleave short decision hops (L=32)
    # with long extraction hops (L=128), using P2 marginals
    marg = load_p2_marginals()
    rng = np.random.default_rng(42)
    mixed_results = {}
    for pattern_name, pattern in [
        ("uniform_extraction", [128]),
        ("mixed_1dec_4ext", [32, 128, 128, 128, 128]),
        ("mixed_2dec_3ext", [32, 32, 128, 128, 128]),
    ]:
        D = len(pattern)
        n_sim = 10000
        for tag in ("dg", "ar3b"):
            key = f"{tag}_L{pattern[0]}"
            if key not in marg:
                continue
            chain_lats = np.zeros(n_sim)
            for hop_L in pattern:
                mk = f"{tag}_L{hop_L}"
                if mk not in marg:
                    continue
                m = marg[mk]
                mu, sigma = fit_lognormal(m["p50"], m["p99"])
                samples = sample_lognormal(mu, sigma, n_sim, rng)
                chain_lats += samples
            mixed_results[f"{pattern_name}_{tag}"] = {
                "D": D,
                "p50_ms": float(np.percentile(chain_lats, 50)),
                "p99_ms": float(np.percentile(chain_lats, 99)),
                "p999_ms": float(np.percentile(chain_lats, 99.9)),
            }
        if f"{pattern_name}_dg" in mixed_results and \
           f"{pattern_name}_ar3b" in mixed_results:
            dg_r = mixed_results[f"{pattern_name}_dg"]
            ar_r = mixed_results[f"{pattern_name}_ar3b"]
            mixed_results[f"{pattern_name}_ratio"] = {
                "p50": round(ar_r["p50_ms"] / dg_r["p50_ms"], 2),
                "p999": round(ar_r["p999_ms"] / dg_r["p999_ms"], 2),
            }
    out["mixed_hop_simulation"] = mixed_results

    # Charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Chart 1: cumulative p50 vs depth (linearity)
    ax = axes[0]
    for label, chain, c in [("DG", dg_chain, "#E8734A"),
                            ("AR-3B", ar_chain, "#4A90D9")]:
        ds = sorted(int(d) for d in chain["per_depth"])
        p50s = [chain["per_depth"][str(d)]["cumulative_latency"]["p50_ms"]
                for d in ds]
        ax.plot(ds, p50s, "o-", color=c, label=f"{label} (measured)")
    ds_arr = np.array(ds)
    dg_slope = dg_chain["per_depth"]["1"]["cumulative_latency"]["p50_ms"]
    ar_slope = ar_chain["per_depth"]["1"]["cumulative_latency"]["p50_ms"]
    ax.plot(ds_arr, ds_arr * dg_slope, "--", color="#E8734A", alpha=0.5,
            label=f"DG linear ({dg_slope:.0f}ms/hop)")
    ax.plot(ds_arr, ds_arr * ar_slope, "--", color="#4A90D9", alpha=0.5,
            label=f"AR linear ({ar_slope:.0f}ms/hop)")
    ax.set_xlabel("chain depth D")
    ax.set_ylabel("cumulative latency p50 (ms)")
    ax.set_title("P-E1: linearity in depth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Chart 2: p99.9 ratio migration (CLT)
    ax = axes[1]
    ax.plot(depths, ratios_p999, "s-", color="red", label="p99.9 ratio (AR/DG)")
    ax.plot(depths, ratios_p50, "o-", color="blue", label="p50 ratio (AR/DG)")
    ax.axhline(ratios_p50[0] if ratios_p50 else 1.9, color="gray", ls=":",
               label=f"mean ratio ({ratios_p50[0]:.2f}x)" if ratios_p50 else "")
    ax.axhline(ratios_p999[0] if ratios_p999 else 1.27, color="red", ls=":",
               alpha=0.3, label=f"D=1 p99.9 ({ratios_p999[0]:.2f}x)"
               if ratios_p999 else "")
    ax.set_xlabel("chain depth D")
    ax.set_ylabel("AR / DG latency ratio")
    ax.set_title("P-E2: tail concentration (CLT)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Chart 3: mixed-hop simulation
    ax = axes[2]
    patterns = ["uniform_extraction", "mixed_1dec_4ext", "mixed_2dec_3ext"]
    labels = ["uniform\n(all L=128)", "1 dec + 4 ext", "2 dec + 3 ext"]
    x = np.arange(len(patterns))
    w = 0.35
    dg_vals = [mixed_results.get(f"{p}_dg", {}).get("p50_ms", 0) for p in patterns]
    ar_vals = [mixed_results.get(f"{p}_ar3b", {}).get("p50_ms", 0) for p in patterns]
    ax.bar(x - w/2, dg_vals, w, color="#E8734A", label="DG")
    ax.bar(x + w/2, ar_vals, w, color="#4A90D9", label="AR-3B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("cumulative p50 (ms)")
    ax.set_title("R1: mixed-hop simulation (D=5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(RESULTS / "chain_clt.png", dpi=150)

    with open(RESULTS / "chain_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
