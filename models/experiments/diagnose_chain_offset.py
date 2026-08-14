#!/usr/bin/env python3
"""
Issue #55 Step 0 — CPU-only decisive test: is the chain quality collapse
(D=1 → D=3, ~0.78 → ~0.46) a gold-misalignment bug in the chain runner?

run_chain_experiment.py never resets `cursor` between depths, but always
slices gold from index 0. So only D=1 is aligned; every later depth scores
predictions against unrelated gold records.

This script:
  1. Loads 2000 saved DG predictions (results/diffusiongemma_preds.jsonl)
  2. Loads 2000 gold records (datasets/splits/test/bgl.jsonl)
  3. Scores all 2000 preds against gold shifted by offsets {0,100,400,900,1900}
  4. Reproduces the exact per-depth scoring from the buggy runner
  5. Computes the empty-prediction floor for reference

If shifted scores land ~0.46 while aligned (offset 0) reproduces ~0.78,
the collapse is the bug — not a model property.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES
from metrics.quality_metrics import field_f1

PREDS_PATH = _REPO_ROOT / "results" / "diffusiongemma_preds.jsonl"
GOLD_PATH = _REPO_ROOT / "datasets" / "splits" / "test" / "bgl.jsonl"
N = 2000


def load_gold(n):
    recs = []
    with open(GOLD_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                recs.append(Record(fields=json.loads(line)))
    return recs


def load_preds(n):
    recs = []
    with open(PREDS_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                obj = json.loads(line)
                recs.append(Record(fields=obj["record"]))
    return recs


def score(preds, golds, label):
    res = field_f1(preds, golds)
    macro = res["macro_field_f1"]
    per_field = {n: res[f"f1_{n}"] for n in FIELD_NAMES}
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  macro Field-F1 = {macro:.4f}")
    print(f"  per-field:")
    for n in FIELD_NAMES:
        print(f"    {n:20s} {per_field[n]:.4f}")
    return macro, per_field


def main():
    print("Loading gold and predictions...")
    gold = load_gold(N)
    preds = load_preds(N)
    print(f"  {len(gold)} gold records, {len(preds)} predictions")

    # --- Arm 1: shift all 2000 preds against gold by each offset ---
    print("\n" + "=" * 60)
    print("ARM 1: shifted-gold test (all 2000 preds vs gold[offset:])")
    print("=" * 60)
    offsets = [0, 100, 400, 900, 1900]
    shift_results = {}
    for off in offsets:
        shifted_gold = [gold[(i + off) % N] for i in range(N)]
        macro, per_field = score(preds, shifted_gold,
                                 f"offset={off:>4d}  (all {N} preds)")
        shift_results[off] = {"macro_f1": macro, "per_field": per_field}

    # Check P1: offset 100-1900 all land in 0.43-0.50, spread < 0.02
    misaligned = [shift_results[o]["macro_f1"] for o in offsets[1:]]
    spread = max(misaligned) - min(misaligned)
    print(f"\n  Misaligned scores: {[f'{m:.4f}' for m in misaligned]}")
    print(f"  Spread (max-min): {spread:.4f}  (P1 predicts < 0.02)")

    # --- Arm 2: reproduce exact per-depth scoring from buggy runner ---
    print("\n" + "=" * 60)
    print("ARM 2: reproduce per-depth scoring from buggy runner")
    print("=" * 60)
    # D=1: preds[0:100] vs gold[0:100]
    # D=3: preds[100:400] vs gold[0:300]
    # D=5: preds[400:900] vs gold[0:500]
    # D=10: preds[900:1900] vs gold[0:1000]
    # D=20: preds[1900:2000]+preds[0:1900] vs gold[0:2000]
    depth_configs = [
        (1, 0, 100),
        (3, 100, 400),
        (5, 400, 900),
        (10, 900, 1900),
        (20, 1900, 3900),  # wraps mod 2000
    ]
    published = {1: 0.783, 3: 0.4565, 5: 0.4554, 10: 0.4654, 20: 0.4598}
    print(f"\n  {'D':>3s}  {'reproduced':>10s}  {'published':>10s}  {'match?':>8s}")
    for D, start, end in depth_configs:
        depth_preds = [preds[i % N] for i in range(start, end)]
        depth_golds = gold[:len(depth_preds)]
        res = field_f1(depth_preds, depth_golds)
        macro = res["macro_field_f1"]
        pub = published[D]
        match = "YES" if abs(macro - pub) < 0.005 else "no"
        print(f"  {D:>3d}  {macro:>10.4f}  {pub:>10.4f}  {match:>8s}")

    # --- Arm 3: empty-prediction floor (P5) ---
    print("\n" + "=" * 60)
    print("ARM 3: empty-prediction floor")
    print("=" * 60)
    empty_preds = [Record(fields={}) for _ in range(N)]
    res_empty = field_f1(empty_preds, gold)
    empty_floor = res_empty["macro_field_f1"]
    per_field_empty = {n: res_empty[f"f1_{n}"] for n in FIELD_NAMES}
    print(f"  empty-prediction macro F1 = {empty_floor:.4f}")
    print(f"  per-field (fields that score > 0 with empty pred):")
    for n in FIELD_NAMES:
        if per_field_empty[n] > 0.001:
            print(f"    {n:20s} {per_field_empty[n]:.4f}")

    # --- Arm 4: aligned per-depth (the CORRECT scoring) ---
    print("\n" + "=" * 60)
    print("ARM 4: ALIGNED per-depth scoring (the fix — preds[i] vs gold[i])")
    print("=" * 60)
    for D, start, end in depth_configs:
        depth_preds = [preds[i % N] for i in range(start, end)]
        depth_golds = [gold[i % N] for i in range(start, end)]
        res = field_f1(depth_preds, depth_golds)
        aligned_macro = res["macro_field_f1"]
        print(f"  D={D:>3d}  aligned F1 = {aligned_macro:.4f}  (n={len(depth_preds)})")

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    aligned = shift_results[0]["macro_f1"]
    misaligned_avg = sum(misaligned) / len(misaligned)
    print(f"  Aligned (offset=0):     {aligned:.4f}  (published D=1: 0.783)")
    print(f"  Misaligned (avg):       {misaligned_avg:.4f}  (published D≥3: ~0.46)")
    print(f"  Empty-prediction floor: {empty_floor:.4f}")
    print(f"  Misaligned > floor?     {misaligned_avg > empty_floor}  "
          f"(P5: misaligned ~0.46 exceeds empty floor)")
    if abs(aligned - 0.783) < 0.01 and 0.43 < misaligned_avg < 0.50:
        print("\n  ==> P1 CONFIRMED: the collapse is a gold-misalignment bug.")
        print("      D=1 quality (0.783) is the only trustworthy quality row.")
        print("      The D≥3 plateau (~0.46) is the random-pairing floor.")
    else:
        print("\n  ==> P1 NOT clearly confirmed — investigate further.")


if __name__ == "__main__":
    main()
