#!/usr/bin/env python3
"""
Issue #53 — compute the semantic-field gold-rule ceiling.

The three "semantic" fields are deterministic functions of the same regex
that produces the structural gold (datasets/prepare.py:317-334):

  event_type       = message_template[:30] or "unknown"   (truncation of structural)
  remediation_hint = "none"                                 (constant string)
  affected_entity  = ENTITY_RE.search(content)             (regex span extraction)

None requires inference. The gold-generating rule scores 1.000 by construction.
This script:
  1. Verifies the rule reproduces the gold (ceiling = 1.000)
  2. Shows value distributions (to prove they're not semantic)
  3. Compares model scores against the ceiling
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES
from metrics.quality_metrics import field_f1
from models.baselines.ar_wrapper import _dict_to_record

GOLD_PATH = _REPO_ROOT / "datasets" / "splits" / "test" / "bgl.jsonl"
PREDS_PATH = _REPO_ROOT / "results" / "diffusiongemma_preds.jsonl"
ENTITY_RE = re.compile(r"(blk_-?\d+|node\d+|R\d+-M\d+-N\d+|tbird-\d+|spirit-\d+|task\d+)")

SEMANTIC_FIELDS = ["event_type", "affected_entity", "remediation_hint"]
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


def recompute_semantic(gold):
    """Re-apply the gold-generating rules to each gold record's structural fields."""
    recomputed = []
    for rec in gold:
        mt = rec.get("message_template") or ""
        content = mt  # In gold, we don't have the raw content; use message_template
        event_type = mt[:30] or "unknown"
        remediation_hint = "none"
        ent_m = ENTITY_RE.search(content)
        affected_entity = ent_m.group(1) if ent_m else ""
        r = Record(fields=dict(rec.fields))
        r.set("event_type", event_type)
        r.set("remediation_hint", remediation_hint)
        r.set("affected_entity", affected_entity)
        recomputed.append(r)
    return recomputed


def main():
    gold = load_gold(N)
    preds = load_preds(N)
    print(f"Loaded {len(gold)} gold records, {len(preds)} DG predictions")

    # ─── 1. Verify the gold-generating rule reproduces gold (ceiling = 1.000) ─
    print("\n" + "=" * 60)
    print("1. GOLD-RULE CEILING: re-apply gold-generating rules to gold records")
    print("=" * 60)
    recomputed = recompute_semantic(gold)
    res = field_f1(recomputed, gold)
    print(f"  Ceiling (gold-rule vs gold): macro F1 = {res['macro_field_f1']:.4f}")
    for f in SEMANTIC_FIELDS:
        print(f"    {f:22s}  ceiling = {res[f'f1_{f}']:.4f}")

    # ─── 2. Value distributions (prove they're not semantic) ──────────────
    print("\n" + "=" * 60)
    print("2. VALUE DISTRIBUTIONS in gold (first 2000 BGL test events)")
    print("=" * 60)
    for f in SEMANTIC_FIELDS:
        vals = [str(rec.get(f, "")) for rec in gold]
        counts = Counter(vals)
        n_unique = len(counts)
        top5 = counts.most_common(5)
        print(f"\n  {f}: {n_unique} unique values in {len(vals)} events")
        for val, cnt in top5:
            print(f"    '{val[:50]}'  ×{cnt}  ({100*cnt/len(vals):.1f}%)")

    # ─── 3. Model scores vs ceiling ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. MODEL SCORES vs CEILING")
    print("=" * 60)
    res_model = field_f1(preds, gold)
    print(f"\n  {'field':22s}  {'ceiling':>8s}  {'DG model':>8s}  {'gap':>8s}")
    print("  " + "-" * 50)
    for f in SEMANTIC_FIELDS:
        ceiling = res[f"f1_{f}"]
        model = res_model[f"f1_{f}"]
        gap = ceiling - model
        print(f"  {f:22s}  {ceiling:>8.4f}  {model:>8.4f}  {gap:>+8.4f}")

    # ─── 4. What the models are actually doing ────────────────────────────
    print("\n" + "=" * 60)
    print("4. WHAT MODELS ARE ACTUALLY DOING (DG predictions)")
    print("=" * 60)
    for f in SEMANTIC_FIELDS:
        pred_vals = [str(rec.get(f, "")) for rec in preds]
        gold_vals = [str(rec.get(f, "")) for rec in gold]
        exact_match = sum(1 for p, g in zip(pred_vals, gold_vals) if p == g)
        print(f"\n  {f}:")
        print(f"    exact match: {exact_match}/{N} ({100*exact_match/N:.1f}%)")
        # Show a few mismatches
        mismatches = [(p, g) for p, g in zip(pred_vals, gold_vals) if p != g]
        if mismatches:
            print(f"    sample mismatches (pred | gold):")
            for p, g in mismatches[:3]:
                print(f"      '{p[:40]}' | '{g[:40]}'")

    # ─── 5. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print(f"  event_type:       gold-rule ceiling = {res['f1_event_type']:.4f}  "
          f"(truncation of message_template[:30])")
    print(f"  affected_entity:  gold-rule ceiling = {res['f1_affected_entity']:.4f}  "
          f"(regex span extraction)")
    print(f"  remediation_hint: gold-rule ceiling = {res['f1_remediation_hint']:.4f}  "
          f"(constant 'none')")
    print()
    print("  All three 'semantic' fields are deterministic functions of the same")
    print("  regex/structure that produces the structural gold. No inference is")
    print("  required. The reachable ceiling is rule-imitation, not quality.")
    print()
    print("  drain-only scores 0.998 on these fields (per_field_report.md) —")
    print("  a regex parser cannot out-infer an LLM; it can only re-derive the")
    print("  rule that wrote the labels.")


if __name__ == "__main__":
    main()
