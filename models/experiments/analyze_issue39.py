#!/usr/bin/env python3
"""Issue #39 — why is DM more perturbation-sensitive than AR?

Analyzes existing perturbation results (from #27) to test 5 proposed
mechanisms. This is a CPU-only analysis script — no GPU needed.

Mechanisms:
  M1: AR's left-to-right conditioning protects against noise back-propagation
  M2: LLaDA's denoising loop inherits inpainting behavior (regression to mean)
  M3: LLaDA-8B is simply a weaker model (lower clean F1 → larger relative drops)
  M4: DM's parallel decoding has no error-correction path (can't "read ahead")
  M5: The 12-field schema is copy-heavy; perturbations break copy patterns

Tests:
  T1 (M3): Normalize F1 drop by clean F1 — does the gap persist?
  T2 (M1/M4): Per-field drop analysis — which fields are most affected?
  T3 (M2): Check if FIELD_REORDER (semantic, not copy) hurts DM more than copy perturbations
  T4 (M5): Classify perturbations as copy-breaking vs semantic, compare DM/AR ratio
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import FIELD_NAMES, FieldCategory, FIELD_BY_NAME


def load_results(path):
    with open(path) as f:
        return json.load(f)


def classify_perturbation(name):
    """Classify a perturbation as copy-breaking or semantic."""
    copy_breaking = {"whitespace", "separator_change", "case_variation",
                     "encoding_noise", "wrapping_noise"}
    semantic = {"field_reorder", "abbreviation", "synonym_substitution",
                "timestamp_format", "missing_field"}
    base = name.split("_i")[0]
    if base in copy_breaking:
        return "copy_breaking"
    return "semantic"


def t1_normalize_by_clean(ar, dm):
    """T1 (M3): Normalize F1 drop by clean F1."""
    ar_clean = ar["categories"]["clean"]["field_f1"]
    dm_clean = dm["categories"]["clean"]["field_f1"]

    print("\n=== T1: Normalized F1 drop (M3: weaker model?) ===")
    print(f"  AR clean F1: {ar_clean:.3f}  DM clean F1: {dm_clean:.3f}")
    print(f"  AR is {ar_clean/dm_clean:.2f}x better on clean data")
    print()

    print(f"  {'perturbation':>30s}  {'AR drop':>8s}  {'DM drop':>8s}  "
          f"{'DM/AR':>6s}  {'AR norm%':>9s}  {'DM norm%':>9s}")
    print("  " + "-" * 80)

    for key in sorted(ar["categories"].keys()):
        if key == "clean":
            continue
        if key not in dm["categories"]:
            continue
        ar_drop = ar["categories"][key].get("f1_drop", 0)
        dm_drop = dm["categories"][key].get("f1_drop", 0)
        if ar_drop == 0:
            continue
        ratio = dm_drop / ar_drop
        ar_norm = ar_drop / ar_clean * 100
        dm_norm = dm_drop / dm_clean * 100
        print(f"  {key:>30s}  {ar_drop:+.3f}  {dm_drop:+.3f}  "
              f"{ratio:6.2f}  {ar_norm:8.1f}%  {dm_norm:8.1f}%")

    print("\n  If M3 holds (DM is just weaker), normalized drops should converge.")
    print("  If M3 is refuted, DM drops remain larger even after normalization.")


def t2_per_field(ar_perfield, dm_perfield):
    """T2 (M1/M4): Per-field drop analysis."""
    print("\n=== T2: Per-field F1 drop (M1/M4: which fields are most affected?) ===")

    ar_clean = ar_perfield["categories"]["clean"]
    dm_clean = dm_perfield["categories"]["clean"]

    print(f"\n  {'field':>25s}  {'AR clean':>8s}  {'DM clean':>8s}  "
          f"{'AR drop':>8s}  {'DM drop':>8s}  {'DM/AR':>6s}")
    print("  " + "-" * 75)

    # Use intensity=1.0 for the worst case
    for ptype in ["case_variation", "field_reorder", "timestamp_format"]:
        key = f"{ptype.lower()}_i1.0"
        if key not in ar_perfield["categories"] or key not in dm_perfield["categories"]:
            continue
        ar_pert = ar_perfield["categories"][key]
        dm_pert = dm_perfield["categories"][key]

        print(f"\n  --- {ptype} (i=1.0) ---")
        for field in FIELD_NAMES:
            f1_key = f"f1_{field}"
            ar_c = ar_clean.get(f1_key, 0)
            dm_c = dm_clean.get(f1_key, 0)
            ar_p = ar_pert.get(f1_key, 0)
            dm_p = dm_pert.get(f1_key, 0)
            ar_drop = ar_c - ar_p
            dm_drop = dm_c - dm_p
            ratio = dm_drop / ar_drop if ar_drop > 0.001 else float("inf")
            if abs(ar_drop) > 0.001 or abs(dm_drop) > 0.001:
                print(f"  {field:>25s}  {ar_c:8.3f}  {dm_c:8.3f}  "
                      f"{ar_drop:+.3f}  {dm_drop:+.3f}  {ratio:6.1f}")


def t3_t4_copy_vs_semantic(ar, dm):
    """T3/T4: Compare DM/AR ratio for copy-breaking vs semantic perturbations."""
    print("\n=== T3/T4: Copy-breaking vs Semantic perturbations ===")

    copy_ratios = []
    semantic_ratios = []

    for key in sorted(ar["categories"].keys()):
        if key == "clean":
            continue
        if key not in dm["categories"]:
            continue
        ar_drop = ar["categories"][key].get("f1_drop", 0)
        dm_drop = dm["categories"][key].get("f1_drop", 0)
        if ar_drop <= 0:
            continue
        ratio = dm_drop / ar_drop
        ptype = classify_perturbation(key)
        if ptype == "copy_breaking":
            copy_ratios.append((key, ratio))
        else:
            semantic_ratios.append((key, ratio))

    import statistics as st
    copy_avg = st.mean(r for _, r in copy_ratios) if copy_ratios else 0
    sem_avg = st.mean(r for _, r in semantic_ratios) if semantic_ratios else 0

    print(f"\n  Copy-breaking perturbations (avg DM/AR ratio): {copy_avg:.2f}x")
    for k, r in sorted(copy_ratios, key=lambda x: -x[1]):
        print(f"    {k:35s}  {r:.2f}x")

    print(f"\n  Semantic perturbations (avg DM/AR ratio): {sem_avg:.2f}x")
    for k, r in sorted(semantic_ratios, key=lambda x: -x[1]):
        print(f"    {k:35s}  {r:.2f}x")

    print(f"\n  If M2 holds (inpainting regression), semantic perturbations")
    print(f"  should hurt DM MORE than copy-breaking ones.")
    print(f"  Copy-breaking avg: {copy_avg:.2f}x  Semantic avg: {sem_avg:.2f}x")
    if sem_avg > copy_avg:
        print(f"  → Semantic perturbations hurt DM more → M2 SUPPORTED")
    else:
        print(f"  → Copy-breaking perturbations hurt DM more → M2 REFUTED")


def main():
    ar = load_results("results/perturbation_ar3b.json")
    dm = load_results("results/perturbation_dm_hf.json")

    ar_pf = load_results("results/perturbation_ar3b_perfield.json")
    dm_pf = load_results("results/perturbation_llada_perfield.json")

    print("=" * 80)
    print("Issue #39: Why is DM more perturbation-sensitive than AR?")
    print("=" * 80)

    t1_normalize_by_clean(ar, dm)
    t2_per_field(ar_pf, dm_pf)
    t3_t4_copy_vs_semantic(ar, dm)

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    ar_clean = ar["categories"]["clean"]["field_f1"]
    dm_clean = dm["categories"]["clean"]["field_f1"]

    import statistics as st
    ratios = []
    for key in ar["categories"]:
        if key == "clean" or key not in dm["categories"]:
            continue
        ar_drop = ar["categories"][key].get("f1_drop", 0)
        dm_drop = dm["categories"][key].get("f1_drop", 0)
        if ar_drop > 0:
            ratios.append(dm_drop / ar_drop)

    avg_ratio = st.mean(ratios) if ratios else 0
    print(f"\n  Average DM/AR drop ratio: {avg_ratio:.2f}x")
    print(f"  Clean F1 ratio (AR/DM): {ar_clean/dm_clean:.2f}x")
    print(f"  AR is {ar_clean/dm_clean:.2f}x better on clean, DM drops {avg_ratio:.2f}x more")

    if avg_ratio > ar_clean / dm_clean:
        print(f"\n  M3 REFUTED: DM's sensitivity is NOT explained by lower clean F1.")
        print(f"  The drop ratio ({avg_ratio:.2f}x) exceeds the clean gap ({ar_clean/dm_clean:.2f}x).")
    else:
        print(f"\n  M3 SUPPORTED: DM's sensitivity is proportional to its lower clean F1.")


if __name__ == "__main__":
    main()
