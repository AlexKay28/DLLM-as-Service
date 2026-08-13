#!/usr/bin/env python3
"""
Issue #19 (App C) analysis: consistency-vs-width degradation curves.

Reads wide_schema results for both DG and AR-3B, tests:
  P-C1: latency ratio tracks ~12 tokens/field (DG stepwise, AR linear)
  P-C2: DG consistency degrades monotonically with F, faster than AR
  P-C3: DG schema validity stays >=95% even as consistency falls
  P-C4: (semi-AR B=2 — not tested here, noted)

Outputs results/wide_schema_analysis.json + results/wide_schema.{png,html}.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _REPO_ROOT / "results"


def load(path):
    return json.load(open(path))


def main():
    dg = load(RESULTS / "wide_schema_dg.json")
    ar = load(RESULTS / "wide_schema_ar3b.json")

    out = {"P_C1_latency": {}, "P_C2_consistency": {}, "P_C3_validity": {}}

    levels = sorted(int(l) for l in dg["per_level"])

    # P-C1: latency ratio
    for label, data in [("dg", dg), ("ar", ar)]:
        lats = []
        for L in levels:
            e = data["per_level"][str(L)]
            lats.append({"level": L, "n_fields": e["n_fields"],
                         "p50_ms": e["latency"]["p50_ms"],
                         "mean_tokens": e["mean_output_tokens"]})
        out["P_C1_latency"][label] = lats

    # ratios
    ratios = []
    for L in levels:
        dg_p50 = dg["per_depth"][str(L)]["latency"]["p50_ms"] if False else \
                 dg["per_level"][str(L)]["latency"]["p50_ms"]
        ar_p50 = ar["per_level"][str(L)]["latency"]["p50_ms"]
        ratios.append(round(ar_p50 / dg_p50, 2) if dg_p50 else None)
    out["P_C1_latency"]["ratio_AR_over_DG"] = ratios

    # P-C2: consistency degradation
    for label, data in [("dg", dg), ("ar", ar)]:
        cons = []
        for L in levels:
            e = data["per_level"][str(L)]
            cons.append({"level": L, "n_fields": e["n_fields"],
                         "consistency_pct": e["quality"]["consistency_pct"]})
        out["P_C2_consistency"][label] = cons

    dg_cons = [dg["per_level"][str(L)]["quality"]["consistency_pct"]
               for L in levels]
    ar_cons = [ar["per_level"][str(L)]["quality"]["consistency_pct"]
               for L in levels]
    out["P_C2_consistency"]["dg_monotonic_decrease"] = all(
        dg_cons[i] >= dg_cons[i+1] for i in range(len(dg_cons)-1))
    out["P_C2_consistency"]["dg_degradation_faster"] = (
        (dg_cons[0] - dg_cons[-1]) > (ar_cons[0] - ar_cons[-1]))
    out["P_C2_consistency"]["kill_condition_triggered"] = (
        len(dg_cons) > 2 and dg_cons[2] < 80)

    # P-C3: validity vs consistency decoupling
    for label, data in [("dg", dg), ("ar", ar)]:
        vals = []
        for L in levels:
            e = data["per_level"][str(L)]
            vals.append({"level": L, "validity": e["quality"]["schema_valid_pct"],
                         "consistency": e["quality"]["consistency_pct"]})
        out["P_C3_validity"][label] = vals

    dg_valid = [dg["per_level"][str(L)]["quality"]["schema_valid_pct"]
                for L in levels]
    out["P_C3_validity"]["dg_stays_above_95"] = all(v >= 95 for v in dg_valid)
    out["P_C3_validity"]["decoupled"] = (
        dg_valid[-1] >= 95 and dg_cons[-1] < dg_cons[0] * 0.9)

    # Charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for label, data, c in [("DG", dg, "#E8734A"), ("AR-3B", ar, "#4A90D9")]:
        ns = [data["per_level"][str(L)]["n_fields"] for L in levels]
        ps = [data["per_level"][str(L)]["latency"]["p50_ms"] for L in levels]
        ax.plot(ns, ps, "o-", color=c, label=label)
    ax.set_xlabel("schema width (fields)")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("P-C1: latency vs schema width")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, data, c in [("DG", dg, "#E8734A"), ("AR-3B", ar, "#4A90D9")]:
        ns = [data["per_level"][str(L)]["n_fields"] for L in levels]
        cs = [data["per_level"][str(L)]["quality"]["consistency_pct"] for L in levels]
        ax.plot(ns, cs, "o-", color=c, label=label)
    ax.axhline(80, color="red", ls=":", alpha=0.3, label="kill condition (80%)")
    ax.set_xlabel("schema width (fields)")
    ax.set_ylabel("consistency (%)")
    ax.set_title("P-C2: consistency degrades with width")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for label, data, c in [("DG", dg, "#E8734A"), ("AR-3B", ar, "#4A90D9")]:
        ns = [data["per_level"][str(L)]["n_fields"] for L in levels]
        vs = [data["per_level"][str(L)]["quality"]["schema_valid_pct"] for L in levels]
        ax.plot(ns, vs, "o-", color=c, label=label)
    ax.axhline(95, color="green", ls=":", alpha=0.3, label="95% threshold")
    ax.set_xlabel("schema width (fields)")
    ax.set_ylabel("schema validity (%)")
    ax.set_title("P-C3: validity stays high")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "wide_schema.png", dpi=150)

    with open(RESULTS / "wide_schema_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
