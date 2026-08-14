#!/usr/bin/env python3
"""
Issue #24: Analyze iterative refinement results.

Compares AR-3B (vLLM, full regeneration, deterministic) vs DM (LLaDA-8B,
selective masking, targeted refinement). DM should improve with refinement;
AR should be flat.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ar_path = REPO / "results" / "refinement_ar3b.json"
    dm_path = REPO / "results" / "refinement_dm_hf.json"

    ar = load(ar_path) if ar_path.exists() else None
    dm = load(dm_path) if dm_path.exists() else None

    if not ar or not dm:
        print("Missing AR or DM results.")
        sys.exit(1)

    print("=== Issue #24: Iterative refinement comparison ===\n")
    print(f"{'Model':>12}  {'Round 1':>8}  {'Round 2':>8}  {'Round 3':>8}  {'Gain':>6}")
    print("-" * 55)

    ar_round1 = ar["rounds"].get("round_1", {}).get("field_f1", 0)
    ar_round3 = ar["rounds"].get("round_3", {}).get("field_f1", 0)
    dm_round1 = dm["rounds"].get("round_1", {}).get("field_f1", 0)
    dm_round3 = dm["rounds"].get("round_3", {}).get("field_f1", 0)

    ar_gain = ar_round3 - ar_round1
    dm_gain = dm_round3 - dm_round1

    def print_row(name, rounds):
        vals = [rounds.get(f"round_{i}", {}).get("field_f1", 0) for i in [1, 2, 3]]
        vals = [f"{v:.3f}" for v in vals]
        gain = ""
        if vals[0] != "0.000" and vals[2] != "0.000":
            gain = f"{float(vals[2]) - float(vals[0]):+.3f}"
        print(f"{name:>12}  {vals[0]:>8}  {vals[1]:>8}  {vals[2]:>8}  {gain:>6}")

    print_row("AR-3B", ar["rounds"])
    print_row("DM-8B", dm["rounds"])

    print("\n=== Verdicts ===\n")

    # P-F5: DM F1 improves with refinement
    p_f5 = "PASS" if dm_gain > 0.01 else "REFUTED"
    print(f"P-F5 (DM F1 gain > 0.01): {p_f5}  (gain={dm_gain:+.3f})")

    # P-F6: AR F1 flat with refinement
    p_f6 = "PASS" if abs(ar_gain) < 0.005 else "REFUTED"
    print(f"P-F6 (AR F1 flat): {p_f6}  (gain={ar_gain:+.3f})")

    # DM round 3 > AR round 1?
    p_f9 = "PASS" if dm_round3 > ar_round1 else "REFUTED"
    print(f"P-F9 (DM R3 F1 > AR R1 F1): {p_f9}  "
          f"(DM={dm_round3:.3f}  AR={ar_round1:.3f})")

    # DM refinement is cheaper than full re-extraction
    dm_r1 = dm["rounds"].get("round_1", {})
    dm_r2 = dm["rounds"].get("round_2", {})
    if dm_r1 and dm_r2:
        dm_r1_p50 = dm_r1.get("p50_ms", 0)
        dm_r2_p50 = dm_r2.get("p50_ms", 0)
        p_f10 = "PASS" if dm_r2_p50 < dm_r1_p50 else "REFUTED"
        print(f"P-F10 (DM refinement p50 < extraction p50): {p_f10}  "
              f"(R1={dm_r1_p50:.0f}ms  R2={dm_r2_p50:.0f}ms)")

    report_path = REPO / "results" / "refinement_report.md"
    with open(report_path, "w") as f:
        f.write("# Issue #24: Iterative refinement loops\n\n")
        f.write("| Model | Round 1 F1 | Round 2 F1 | Round 3 F1 | Gain |\n")
        f.write("|-------|-----------|-----------|-----------|------|\n")
        for label, data in [("AR-3B", ar), ("DM-8B", dm)]:
            r1 = data["rounds"].get("round_1", {}).get("field_f1", 0)
            r2 = data["rounds"].get("round_2", {}).get("field_f1", 0)
            r3 = data["rounds"].get("round_3", {}).get("field_f1", 0)
            g = r3 - r1
            f.write(f"| {label} | {r1:.3f} | {r2:.3f} | {r3:.3f} | {g:+.3f} |\n")
        f.write(f"\n## Verdicts\n\n")
        f.write(f"- **P-F5** (DM F1 gain > 0.01): {p_f5} (gain={dm_gain:+.3f})\n")
        f.write(f"- **P-F6** (AR F1 flat): {p_f6} (gain={ar_gain:+.3f})\n")
        f.write(f"- **P-F9** (DM R3 > AR R1): {p_f9} (DM={dm_round3:.3f}, AR={ar_round1:.3f})\n")

    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
