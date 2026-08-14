#!/usr/bin/env python3
"""
Issue #54 — per-rule consistency decomposition.

The consistency metric (_check_consistency in quality_metrics.py:134-172) is
3 hardcoded rules over 4 fields, all sharing `severity`, evaluated
all-or-nothing per record. This script checks each rule independently to
see which rule(s) dominate the AR-vs-DM gap.

Rules:
  R1: severity ∈ {ERROR,FATAL,CRITICAL} ⇒ is_anomaly == true
  R2: severity ∈ {INFO,DEBUG} ⇒ anomaly_score ≤ 0.8;
      severity ∈ {FATAL,CRITICAL} ⇒ anomaly_score ≥ 0.5
  R3: error_code non-empty ⇒ severity != "INFO"

Also confirmed: `component` is read at :147 but NEVER USED (dead variable).
The docstring's "component ↔ error_code" constraint does not exist.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES

DG_PREDS = _REPO_ROOT / "results" / "diffusiongemma_preds.jsonl"
AR_PREDS = _REPO_ROOT / "results" / "diag_vllm_preds.jsonl"


def load_preds(path, n=None):
    recs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                obj = json.loads(line)
                recs.append(Record(fields=obj["record"]))
    return recs


def check_rules(rec):
    """Check each rule independently. Returns (r1_pass, r2_pass, r3_pass)."""
    severity = rec.get("severity")
    is_anomaly = rec.get("is_anomaly")
    anomaly_score = rec.get("anomaly_score")
    error_code = rec.get("error_code")

    # R1: severity ∈ {ERROR,FATAL,CRITICAL} ⇒ is_anomaly == true
    r1_pass = True
    if severity and is_anomaly is not None:
        anomaly_bool = is_anomaly if isinstance(is_anomaly, bool) else str(is_anomaly).lower() == "true"
        if severity in ("ERROR", "FATAL", "CRITICAL") and not anomaly_bool:
            r1_pass = False

    # R2: anomaly_score ↔ severity
    r2_pass = True
    if severity and anomaly_score is not None:
        try:
            score = float(anomaly_score)
            if severity in ("INFO", "DEBUG") and score > 0.8:
                r2_pass = False
            if severity in ("FATAL", "CRITICAL") and score < 0.5:
                r2_pass = False
        except (ValueError, TypeError):
            pass

    # R3: error_code present ⇒ severity != "INFO"
    r3_pass = True
    if error_code and severity == "INFO":
        r3_pass = False

    return (r1_pass, r2_pass, r3_pass)


def decompose(records, label):
    n = len(records)
    r1_fail = r2_fail = r3_fail = 0
    all_pass = 0
    only_r1_fail = only_r2_fail = only_r3_fail = 0
    multi_fail = 0

    for rec in records:
        r1, r2, r3 = check_rules(rec)
        fails = sum(1 for x in (r1, r2, r3) if not x)
        if fails == 0:
            all_pass += 1
        elif fails == 1:
            if not r1: only_r1_fail += 1
            elif not r2: only_r2_fail += 1
            else: only_r3_fail += 1
        else:
            multi_fail += 1
        if not r1: r1_fail += 1
        if not r2: r2_fail += 1
        if not r3: r3_fail += 1

    consistency_pct = 100.0 * all_pass / n if n else 0
    print(f"\n{'='*60}")
    print(f"  {label}  (n={n})")
    print(f"{'='*60}")
    print(f"  Overall consistency: {consistency_pct:.2f}%  ({all_pass}/{n})")
    print(f"\n  Per-rule violation rates:")
    print(f"    R1 (severity→is_anomaly):     {100*r1_fail/n:.1f}% fail  ({r1_fail}/{n})")
    print(f"    R2 (severity→anomaly_score):  {100*r2_fail/n:.1f}% fail  ({r2_fail}/{n})")
    print(f"    R3 (error_code→severity):     {100*r3_fail/n:.1f}% fail  ({r3_fail}/{n})")
    print(f"\n  Violation isolation:")
    print(f"    only R1 fails:  {only_r1_fail}/{n} ({100*only_r1_fail/n:.1f}%)")
    print(f"    only R2 fails:  {only_r2_fail}/{n} ({100*only_r2_fail/n:.1f}%)")
    print(f"    only R3 fails:  {only_r3_fail}/{n} ({100*only_r3_fail/n:.1f}%)")
    print(f"    multiple fail:  {multi_fail}/{n} ({100*multi_fail/n:.1f}%)")

    # Severity distribution (since severity drives all 3 rules)
    from collections import Counter
    sev_counts = Counter(str(rec.get("severity", "")) for rec in records)
    print(f"\n  severity distribution:")
    for sev, cnt in sev_counts.most_common():
        print(f"    {sev:12s}  {cnt:5d}  ({100*cnt/n:.1f}%)")

    return {"n": n, "consistency": consistency_pct,
            "r1_fail": r1_fail, "r2_fail": r2_fail, "r3_fail": r3_fail,
            "all_pass": all_pass}


def main():
    print("Issue #54 — per-rule consistency decomposition")
    print("Rules:")
    print("  R1: severity ∈ {ERROR,FATAL,CRITICAL} ⇒ is_anomaly == true")
    print("  R2: severity ∈ {INFO,DEBUG} ⇒ score ≤ 0.8; {FATAL,CRITICAL} ⇒ score ≥ 0.5")
    print("  R3: error_code non-empty ⇒ severity != INFO")
    print("\nNote: `component` is read at quality_metrics.py:147 but NEVER USED.")
    print("      The docstring's 'component ↔ error_code' constraint does not exist.")

    # DG chain predictions (2000)
    dg_recs = load_preds(DG_PREDS)
    dg_res = decompose(dg_recs, "DG (DiffusionGemma-26B, chain predictions)")

    # AR diagnostic predictions (100)
    ar_recs = load_preds(AR_PREDS)
    ar_res = decompose(ar_recs, "AR (Qwen2.5-3B, diagnostic predictions)")

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: DG vs AR")
    print(f"{'='*60}")
    print(f"  {'metric':30s}  {'DG':>10s}  {'AR':>10s}  {'gap':>10s}")
    print(f"  {'-'*62}")
    for key, label in [("consistency", "consistency %"),
                       ("r1_fail", "R1 fail count"),
                       ("r2_fail", "R2 fail count"),
                       ("r3_fail", "R3 fail count")]:
        dv = dg_res[key]
        av = ar_res[key]
        # Normalize to % for comparison
        if key == "consistency":
            gap = dv - av
            print(f"  {label:30s}  {dv:>9.1f}%  {av:>9.1f}%  {gap:>+9.1f}%")
        else:
            dg_pct = 100 * dv / dg_res["n"]
            ar_pct = 100 * av / ar_res["n"]
            gap = dg_pct - ar_pct
            print(f"  {label:30s}  {dv:>5d} ({dg_pct:.1f}%)  {av:>5d} ({ar_pct:.1f}%)  {gap:>+9.1f}%")

    # Binomial CI for the ablation result (n=30)
    print(f"\n{'='*60}")
    print(f"  BINOMIAL CI FOR ABLATION (n=30)")
    print(f"{'='*60}")
    from math import sqrt
    for B, pct in [(1, 70.0), (2, 96.67), (4, 80.0)]:
        p = pct / 100
        n = 30
        se = sqrt(p * (1 - p) / n)
        ci = 1.96 * se * 100
        print(f"  B={B}: {pct:.1f}%  (95% CI: ±{ci:.1f}%, i.e. {pct-ci:.1f}–{pct+ci:.1f}%)")
    print(f"\n  B=1→B=2 swing: 70.0→96.67% = 21→29 records passing (8 records)")
    print(f"  B=2→B=4 swing: 96.67→80.0% = 29→24 records passing (5 records)")
    print(f"  With n=30, a swing of 5 records is inside the binomial noise.")


if __name__ == "__main__":
    main()
