#!/usr/bin/env python3
"""
Issue #25: Analyze bounded-deadline results.

Compares AR-3B (vLLM, max_tokens varies with deadline) against DM
(LLaDA-8B, K=8, always one canvas). Tests whether DM's "completes at
any budget" claim holds over AR's "truncates to budget."
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ar_path = REPO / "results" / "bounded_deadline_ar3b.json"
    dm_path = REPO / "results" / "bounded_deadline_dm_hf.json"

    ar = load(ar_path) if ar_path.exists() else None
    dm = load(dm_path) if dm_path.exists() else None

    if not ar or not dm:
        print("Missing AR or DM results.")
        sys.exit(1)

    print("=== Issue #25: Bounded-deadline comparison ===\n")

    print("AR-3B (vLLM, max_tokens = D/6.28):")
    print(f"{'D(ms)':>7}  {'F1':>6}  {'Field%':>8}  {'Record%':>8}  {'p50ms':>7}")
    for c in ar["cells"]:
        print(f"{c['deadline_ms']:7d}  {c['field_f1']:.3f}  "
              f"{c.get('field_completeness_pct', 0):7.1f}%  "
              f"{c.get('record_completeness_pct', 0):7.1f}%  "
              f"{c['p50_ms']:7.0f}")

    print(f"\nDM (LLaDA-8B, K=8, one canvas):")
    dm_cell = dm["cells"][0]
    print(f"  F1={dm_cell['field_f1']:.3f}  "
          f"Field={dm_cell['field_completeness_pct']:.1f}%  "
          f"Record={dm_cell['record_completeness_pct']:.1f}%  "
          f"p50={dm_cell['p50_ms']:.0f}ms")

    print("\n=== Verdicts ===\n")

    # P-F7: at D=1500, DM record completeness > AR record completeness
    dm_complete = dm_cell.get("record_completeness_pct", 100)
    d1500 = next((c for c in ar["cells"] if c["deadline_ms"] == 1500), None)
    ar_complete_d1500 = d1500.get("record_completeness_pct", 0) if d1500 else 0
    p_f7 = "PASS" if dm_complete > ar_complete_d1500 else "REFUTED"
    print(f"P-F7 (DM record completeness > AR at D=1500): {p_f7}")
    print(f"  DM={dm_complete:.1f}%  AR={ar_complete_d1500:.1f}%")

    # P-F8: AR F1 collapses below D=1000
    ar_d1000 = next((c for c in ar["cells"] if c["deadline_ms"] == 1000), None)
    ar_d200 = next((c for c in ar["cells"] if c["deadline_ms"] == 200), None)
    p_f8 = "PASS" if (ar_d200 and ar_d1000 and
                       ar_d200["field_f1"] < 0.5 * ar_d1000["field_f1"]) else "REFUTED"
    print(f"P-F8 (AR F1 collapses below D=1000): {p_f8}")
    print(f"  D=200 F1={ar_d200['field_f1']:.3f}  D=1000 F1={ar_d1000['field_f1']:.3f}")

    report_path = REPO / "results" / "bounded_deadline_report.md"
    with open(report_path, "w") as f:
        f.write("# Issue #25: Bounded-deadline completion\n\n")
        f.write("## AR-3B (vLLM, max_tokens = D/6.28)\n\n")
        f.write("| D(ms) | F1 | Field% | Record% | p50ms |\n")
        f.write("|-------|----|--------|---------|-------|\n")
        for c in ar["cells"]:
            f.write(f"| {c['deadline_ms']} | {c['field_f1']:.3f} | "
                   f"{c.get('field_completeness_pct', 0):.1f}% | "
                   f"{c.get('record_completeness_pct', 0):.1f}% | "
                   f"{c['p50_ms']:.0f} |\n")
        f.write(f"\n## DM (LLaDA-8B, K=8, one canvas)\n\n")
        f.write(f"- F1: {dm_cell['field_f1']:.3f}\n")
        f.write(f"- Field completeness: {dm_cell['field_completeness_pct']:.1f}%\n")
        f.write(f"- Record completeness: {dm_cell['record_completeness_pct']:.1f}%\n")
        f.write(f"- p50: {dm_cell['p50_ms']:.0f}ms\n")
        f.write(f"\n## Verdicts\n\n")
        f.write(f"- **P-F7** (DM record completeness > AR at D=1500): {p_f7} "
                f"(DM={dm_complete:.1f}%, AR={ar_complete_d1500:.1f}%)\n")
        f.write(f"- **P-F8** (AR F1 collapses below D=1000): {p_f8} "
                f"(D=200 F1={ar_d200['field_f1']:.3f}, D=1000 F1={ar_d1000['field_f1']:.3f})\n")

    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
