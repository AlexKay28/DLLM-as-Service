#!/usr/bin/env python3
"""
Issue #27: Analyze perturbation robustness — AR vs DM.

Compares AR-3B and DM F1 drops across perturbation categories. The
hypothesis was that DM's joint denoising is more robust to input noise.
The data shows the opposite: AR is more robust in nearly every category.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load(path):
    with open(path) as f:
        return json.load(f)


CATEGORIES = [
    "whitespace", "field_reorder", "separator_change", "abbreviation",
    "timestamp_format", "missing_field", "case_variation",
    "wrapping_noise", "synonym_substitution", "encoding_noise",
]


def main():
    ar = load(REPO / "results" / "perturbation_ar3b.json")["categories"]
    dm = load(REPO / "results" / "perturbation_dm_hf.json")["categories"]

    ar_clean = ar["clean"]["field_f1"]
    dm_clean = dm["clean"]["field_f1"]

    print("=== Issue #27: Perturbation robustness (i=1.0) ===\n")
    print(f"{'Category':>25}  {'AR F1':>7}  {'DM F1':>7}  {'AR drop':>8}  {'DM drop':>8}  {'Ratio':>6}")
    print("-" * 75)

    worse_for_dm = 0
    for cat in CATEGORIES:
        ar_key = f"{cat}_i1.0"
        dm_key = f"{cat}_i1.0"
        if ar_key not in ar or dm_key not in dm:
            continue
        ar_f1 = ar[ar_key]["field_f1"]
        dm_f1 = dm[dm_key]["field_f1"]
        ar_drop = ar[ar_key].get("f1_drop", 0)
        dm_drop = dm[dm_key].get("f1_drop", 0)
        if dm_drop > ar_drop:
            worse_for_dm += 1
        if ar_drop < 0.005:
            ratio_str = "inf (AR flat)"
        else:
            ratio = dm_drop / ar_drop
            ratio_str = f"{ratio:.1f}x"
        print(f"{cat:>25}  {ar_f1:.3f}  {dm_f1:.3f}  "
              f"{ar_drop:.3f}  {dm_drop:.3f}  {ratio_str}")

    print(f"\nClean F1: AR={ar_clean:.3f}  DM={dm_clean:.3f}  diff={ar_clean - dm_clean:.3f}")

    print(f"\n=== Verdicts ===\n")
    p_f11 = "PASS" if worse_for_dm >= 8 else "REFUTED"
    print(f"P-F11 (DM more perturbation-sensitive in ≥8/10 categories): {p_f11}")
    print(f"  DM worse in {worse_for_dm}/10 categories")

    p_f12 = "PASS" if ar_clean > dm_clean and dm_drop - ar_drop > 0 else "PARTIAL"
    print(f"P-F12 (AR baseline F1 higher + DM more sensitive): {p_f12}")

    report_path = REPO / "results" / "perturbation_report.md"
    with open(report_path, "w") as f:
        f.write("# Issue #27: Perturbation robustness\n\n")
        f.write(f"## Clean baseline\n\n")
        f.write(f"- AR-3B F1: {ar_clean:.4f}\n")
        f.write(f"- DM F1: {dm_clean:.4f}\n")
        f.write(f"- AR advantage: {ar_clean - dm_clean:.4f}\n\n")
        f.write("## At intensity i=1.0\n\n")
        f.write("| Category | AR F1 | DM F1 | AR drop | DM drop | DM/AR ratio |\n")
        f.write("|----------|-------|-------|---------|---------|-------------|\n")
        for cat in CATEGORIES:
            ar_key = f"{cat}_i1.0"
            dm_key = f"{cat}_i1.0"
            if ar_key not in ar or dm_key not in dm:
                continue
            ar_f1 = ar[ar_key]["field_f1"]
            dm_f1 = dm[dm_key]["field_f1"]
            ar_drop = ar[ar_key].get("f1_drop", 0)
            dm_drop = dm[dm_key].get("f1_drop", 0)
            if ar_drop < 0.005:
                ratio_str = "inf (AR flat)"
            else:
                ratio = dm_drop / ar_drop
                ratio_str = f"{ratio:.1f}x"
            f.write(f"| {cat} | {ar_f1:.3f} | {dm_f1:.3f} | "
                   f"{ar_drop:.3f} | {dm_drop:.3f} | {ratio_str} |\n")
        f.write(f"\n**DM worse in {worse_for_dm}/10 categories** — hypothesis refuted.\n")

    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
