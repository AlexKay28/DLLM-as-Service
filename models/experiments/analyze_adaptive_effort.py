#!/usr/bin/env python3
"""
Issue #24: Analyze adaptive-effort results.

Compares per-bin (routine/medium/novel) F1 and latency between
dInfer (threshold decoder) and AR-3B (vLLM). Tests whether DM's
threshold decoder adapts effort to difficulty (low K on easy, high K
on hard) while AR is flat.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    results_dir = REPO / "results"
    ar_path = results_dir / "adaptive_effort_ar3b.json"
    dinfer_path = results_dir / "adaptive_effort_dinfer.json"
    mdm_path = results_dir / "adaptive_effort_mdm_hf.json"

    ar = load(ar_path) if ar_path.exists() else None
    dinfer = load(dinfer_path) if dinfer_path.exists() else None
    mdm = load(mdm_path) if mdm_path.exists() else None

    bins = ["routine", "medium", "novel"]

    print("=== Issue #24: Adaptive-effort per-bin comparison ===\n")

    models = {"AR-3B (vLLM)": ar, "dInfer (threshold)": dinfer, "DM K=8 (HF)": mdm}
    models = {k: v for k, v in models.items() if v is not None}

    if not models:
        print("No results found.")
        sys.exit(1)

    # Table: per-bin F1
    print(f"{'Model':>22}  {'Bin':>8}  {'N':>4}  {'F1':>6}  "
          f"{'p50ms':>7}  {'p99ms':>7}  {'mean':>7}")
    print("-" * 70)
    for label, data in models.items():
        for b in bins:
            if b in data.get("per_bin", {}):
                r = data["per_bin"][b]
                print(f"{label:>22}  {b:>8}  {r['n']:4d}  {r['field_f1']:.3f}  "
                      f"{r['p50_ms']:7.0f}  {r['p99_ms']:7.0f}  {r['mean_ms']:7.0f}")
        print()

    # Analysis: does DM adapt?
    print("=== Verdicts ===\n")

    if dinfer and ar:
        d_bins = dinfer.get("per_bin", {})
        a_bins = ar.get("per_bin", {})

        # P-F1: DM F1 higher on novel (harder) events
        d_novel_f1 = d_bins.get("novel", {}).get("field_f1", 0)
        a_novel_f1 = a_bins.get("novel", {}).get("field_f1", 0)
        p_f1 = "PASS" if d_novel_f1 > a_novel_f1 else "REFUTED"
        print(f"P-F1 (DM F1 > AR F1 on novel): {p_f1}")
        print(f"  DM novel F1={d_novel_f1:.3f}  AR novel F1={a_novel_f1:.3f}")

        # P-F2: DM latency lower on routine (easier) events
        d_routine_p50 = d_bins.get("routine", {}).get("p50_ms", 0)
        a_routine_p50 = a_bins.get("routine", {}).get("p50_ms", 0)
        p_f2 = "PASS" if d_routine_p50 < a_routine_p50 else "REFUTED"
        print(f"P-F2 (DM p50 < AR p50 on routine): {p_f2}")
        print(f"  DM routine p50={d_routine_p50:.0f}ms  AR routine p50={a_routine_p50:.0f}ms")

        # P-F3: DM latency variance across bins (adaptivity)
        d_lats = [d_bins.get(b, {}).get("mean_ms", 0) for b in bins]
        a_lats = [a_bins.get(b, {}).get("mean_ms", 0) for b in bins]
        d_range = max(d_lats) - min(d_lats) if d_lats else 0
        a_range = max(a_lats) - min(a_lats) if a_lats else 0
        p_f3 = "PASS" if d_range > a_range else "REFUTED"
        print(f"P-F3 (DM latency range > AR range across bins): {p_f3}")
        print(f"  DM range={d_range:.0f}ms  AR range={a_range:.0f}ms")

        # P-F4: DM F1 degrades less from routine to novel
        d_f1_drop = d_bins.get("routine", {}).get("field_f1", 0) - d_bins.get("novel", {}).get("field_f1", 0)
        a_f1_drop = a_bins.get("routine", {}).get("field_f1", 0) - a_bins.get("novel", {}).get("field_f1", 0)
        p_f4 = "PASS" if d_f1_drop < a_f1_drop else "REFUTED"
        print(f"P-F4 (DM F1 drop routine->novel < AR drop): {p_f4}")
        print(f"  DM drop={d_f1_drop:.3f}  AR drop={a_f1_drop:.3f}")

    # Write report
    report_path = REPO / "results" / "adaptive_effort_report.md"
    with open(report_path, "w") as f:
        f.write("# Issue #24: Adaptive-effort via threshold decoding\n\n")
        f.write("## Per-bin results\n\n")
        f.write(f"| Model | Bin | N | F1 | p50ms | p99ms | mean_ms |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for label, data in models.items():
            for b in bins:
                if b in data.get("per_bin", {}):
                    r = data["per_bin"][b]
                    f.write(f"| {label} | {b} | {r['n']} | {r['field_f1']:.3f} | "
                           f"{r['p50_ms']:.0f} | {r['p99_ms']:.0f} | {r['mean_ms']:.0f} |\n")
            f.write("\n")

        if dinfer and ar:
            f.write("## Verdicts\n\n")
            f.write(f"- **P-F1** (DM F1 > AR F1 on novel): {p_f1} "
                    f"(DM={d_novel_f1:.3f}, AR={a_novel_f1:.3f})\n")
            f.write(f"- **P-F2** (DM p50 < AR p50 on routine): {p_f2} "
                    f"(DM={d_routine_p50:.0f}ms, AR={a_routine_p50:.0f}ms)\n")
            f.write(f"- **P-F3** (DM latency range > AR across bins): {p_f3} "
                    f"(DM={d_range:.0f}ms, AR={a_range:.0f}ms)\n")
            f.write(f"- **P-F4** (DM F1 drop routine->novel < AR): {p_f4} "
                    f"(DM={d_f1_drop:.3f}, AR={a_f1_drop:.3f})\n")

    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
