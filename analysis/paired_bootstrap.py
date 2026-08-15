#!/usr/bin/env python3
"""
Issue #75 — Paired bootstrap CI for the AR-3B vs DiffusionGemma equivalence claim.

The paper claims AR-3B and DiffusionGemma-26B (DG) have a 9-field macro F1
difference of 0.0002 (0.8516 vs 0.8518) at n=300, with no confidence interval.
Both models are deterministic at temperature 0 and were evaluated on the SAME
300 BGL events (verified by matching timestamps in the per-event prediction
files), so a PAIRED bootstrap is the correct tool.

Per-event 9-field F1 for event i is the mean of the 9 single-field token-F1
scores (field_f1_single) over the NINE_FIELD_NAMES set. Because every field is
scored on all 300 events, the mean of these per-event scores EXACTLY equals the
reported macro_field_f1_9 = (1/9) * sum_f (1/n) * sum_i f1_{f,i}. The script
asserts this reproduction (round(mean, 4) == reported 0.8516 / 0.8518) before
bootstrapping.

Provenance of the data (exact file paths + keys):
  - gold:          datasets/splits/test/bgl.jsonl  (first 300 records; same
                    order both arms were generated in)
  - AR-3B preds:   results/issue43/ar3b_L256_preds.jsonl  (key "raw", n=300)
  - DG    preds:   results/issue43/dg_L256_preds.jsonl     (key "raw", n=300)
  - aggregated:    results/issue43_nine_field_f1.json
                    (ar3b_L256.field_f1_9 = 0.8516, dg_L256.field_f1_9 = 0.8518)
  - parser:        parse_first_obj  (first-object-wins, #57 R5)
  - dict->Record:  _dict_to_record  (metrics/schema + models/baselines/ar_wrapper)
  - field F1:      field_f1_single   (metrics/quality_metrics)
  - 9 fields:      NINE_FIELD_NAMES  (metrics/quality_metrics)

Secondary: AR-1.5B vs DG. AR-1.5B predictions
(results/issue_evidence_ar1b_L512_preds.jsonl) cover only the FIRST 100 events
(at L=512, a different token budget than DG's L=256), so the paired comparison
uses those 100 overlapping events. The DG value 0.8518 is over 300 events; the
DG mean over the first 100 differs and is reported as-is.

Pure Python stdlib (random, statistics, json) per AGENTS.md — no numpy/scipy.
"""

import json
import random
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the repo's own scoring code verbatim so the per-event F1 exactly
# reproduces the reported 0.8516 / 0.8518.
from metrics.quality_metrics import field_f1_single, NINE_FIELD_NAMES
from models.baselines.ar_wrapper import _dict_to_record
from models.experiments.rescore_parsers import parse_first_obj
from metrics.schema import Record

SEED = 42
N_BOOT = 10_000
TOST_MARGIN = 0.01  # ±0.01 F1 equivalence margin

GOLD_PATH = _REPO_ROOT / "datasets" / "splits" / "test" / "bgl.jsonl"
AR3B_PREDS = _REPO_ROOT / "results" / "issue43" / "ar3b_L256_preds.jsonl"
DG_PREDS = _REPO_ROOT / "results" / "issue43" / "dg_L256_preds.jsonl"
AR1B_PREDS = _REPO_ROOT / "results" / "issue_evidence_ar1b_L512_preds.jsonl"
OUT_JSON = _REPO_ROOT / "results" / "issue75_bootstrap_ci.json"

# Reported macro_field_f1_9 (results/issue43_nine_field_f1.json, n=300) for
# verification that the per-event reconstruction reproduces the paper numbers.
REPORTED_AR3B = 0.8516
REPORTED_DG = 0.8518
# AR-1.5B (results/issue_evidence_ar1b_L512.json, n=100, L=512).
REPORTED_AR1B = 0.8679


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


def load_raw_preds(path):
    raws = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                raws.append(obj.get("raw", ""))
    return raws


def per_event_f1_9(raw_preds, gold):
    """Per-event 9-field macro F1 = mean of 9 single-field token-F1 per event.

    Mirrors metrics.quality_metrics.field_f1 exactly: it parses the raw model
    output with the first-object-wins parser, converts to a Record, and scores
    each field with field_f1_single. Mean of the returned list equals
    macro_field_f1_9 (see module docstring).
    """
    n = min(len(raw_preds), len(gold))
    scores = []
    for i in range(n):
        rec = _dict_to_record(parse_first_obj(raw_preds[i]))
        field_scores = []
        for f in NINE_FIELD_NAMES:
            _, _, f1 = field_f1_single(rec.get(f), gold[i].get(f))
            field_scores.append(f1)
        scores.append(sum(field_scores) / len(field_scores))
    return scores


def _percentile(sorted_vals, pct):
    """Linear-interpolation percentile (matches numpy's default 'linear')."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (n - 1)
    lo_idx = int(rank)
    hi_idx = min(lo_idx + 1, n - 1)
    frac = rank - lo_idx
    return sorted_vals[lo_idx] + (sorted_vals[hi_idx] - sorted_vals[lo_idx]) * frac


def bootstrap_paired(a_scores, b_scores, seed, n_boot):
    """Paired bootstrap: resample (a_i, b_i) index together; delta = mean(b)-mean(a)."""
    rng = random.Random(seed)
    n = len(a_scores)
    deltas = []
    for _ in range(n_boot):
        sa = 0.0
        sb = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            sa += a_scores[j]
            sb += b_scores[j]
        deltas.append(sb / n - sa / n)
    deltas.sort()
    return deltas


def ci_from_deltas(deltas):
    lo = _percentile(deltas, 2.5)
    hi = _percentile(deltas, 97.5)
    lo90 = _percentile(deltas, 5.0)
    hi90 = _percentile(deltas, 95.0)
    return lo, hi, lo90, hi90


def analyze(name, a_scores, b_scores, a_label, b_label):
    a_mean = statistics.fmean(a_scores)
    b_mean = statistics.fmean(b_scores)
    delta = b_mean - a_mean
    deltas = bootstrap_paired(a_scores, b_scores, SEED, N_BOOT)
    lo, hi, lo90, hi90 = ci_from_deltas(deltas)
    se = statistics.pstdev(deltas)
    # TOST equivalence verdicts.
    #   - ci95_in_margin: 95% bootstrap CI fully inside [-margin, +margin]
    #     (the check the issue brief asks for).
    #   - tost_classical_90: classical TOST at alpha=0.05 uses the 90% CI
    #     (1 - 2*alpha); equivalence iff 90% CI inside [-margin, +margin].
    ci95_in = (lo >= -TOST_MARGIN) and (hi <= TOST_MARGIN)
    tost90 = (lo90 >= -TOST_MARGIN) and (hi90 <= TOST_MARGIN)
    return {
        "comparison": name,
        "method": "paired_bootstrap",
        "n_events": len(a_scores),
        "arm_a": a_label,
        "arm_b": b_label,
        "mean_a": round(a_mean, 6),
        "mean_b": round(b_mean, 6),
        "delta_mean_b_minus_a": round(delta, 6),
        "ci_95_low": round(lo, 6),
        "ci_95_high": round(hi, 6),
        "ci_90_low": round(lo90, 6),
        "ci_90_high": round(hi90, 6),
        "bootstrap_se_delta": round(se, 6),
        "n_resamples": N_BOOT,
        "seed": SEED,
        "tost_margin": TOST_MARGIN,
        "tost_95ci_in_margin": bool(ci95_in),
        "tost_classical_90ci_equivalent": bool(tost90),
    }


def main():
    # ---- Primary: AR-3B vs DG on the same 300 events (L=256 both) ----
    gold300 = load_gold(300)
    ar3b_raw = load_raw_preds(AR3B_PREDS)
    dg_raw = load_raw_preds(DG_PREDS)
    ar3b_scores = per_event_f1_9(ar3b_raw, gold300)
    dg_scores = per_event_f1_9(dg_raw, gold300)

    ar3b_mean = statistics.fmean(ar3b_scores)
    dg_mean = statistics.fmean(dg_scores)
    print("== Reconstruction check (AR-3B vs DG, n=300) ==")
    print(f"  AR-3B per-event mean = {ar3b_mean:.6f}  (reported {REPORTED_AR3B}, "
          f"round4={round(ar3b_mean, 4)})")
    print(f"  DG    per-event mean = {dg_mean:.6f}  (reported {REPORTED_DG}, "
          f"round4={round(dg_mean, 4)})")
    assert round(ar3b_mean, 4) == REPORTED_AR3B, \
        f"AR-3B reproduction mismatch: {round(ar3b_mean,4)} != {REPORTED_AR3B}"
    assert round(dg_mean, 4) == REPORTED_DG, \
        f"DG reproduction mismatch: {round(dg_mean,4)} != {REPORTED_DG}"
    print("  OK: per-event reconstruction matches reported macro_field_f1_9.\n")

    primary = analyze("AR-3B vs DiffusionGemma-26B (L=256, n=300)",
                      ar3b_scores, dg_scores, "ar3b_L256", "dg_L256")

    # ---- Secondary: AR-1.5B vs DG on the first 100 events ----
    # AR-1.5B preds are L=512, n=100; pair with DG's first 100 events (same gold
    # order, verified by matching timestamps). Different token budget -> flagged.
    gold100 = load_gold(100)
    ar1b_raw = load_raw_preds(AR1B_PREDS)
    dg100_raw = load_raw_preds(DG_PREDS)[:100]
    ar1b_scores = per_event_f1_9(ar1b_raw, gold100)
    dg100_scores = per_event_f1_9(dg100_raw, gold100)
    ar1b_mean = statistics.fmean(ar1b_scores)
    dg100_mean = statistics.fmean(dg100_scores)
    print("== Reconstruction check (AR-1.5B vs DG first-100) ==")
    print(f"  AR-1.5B per-event mean = {ar1b_mean:.6f}  (reported {REPORTED_AR1B}, "
          f"round4={round(ar1b_mean, 4)})")
    print(f"  DG[0:100] per-event mean = {dg100_mean:.6f}  "
          f"(reported 300-event value is 0.8518, not directly comparable)")
    assert round(ar1b_mean, 4) == REPORTED_AR1B, \
        f"AR-1.5B reproduction mismatch: {round(ar1b_mean,4)} != {REPORTED_AR1B}"
    print("  OK: AR-1.5B per-event reconstruction matches reported 0.8679.\n")

    secondary = analyze("AR-1.5B (L=512) vs DiffusionGemma-26B (L=256), first 100 events",
                        ar1b_scores, dg100_scores, "ar1b_L512", "dg_L256[:100]")
    secondary["note"] = ("AR-1.5B at L=512 vs DG at L=256: different token budgets. "
                          "Paired only on the 100 events AR-1.5B was run on.")

    # ---- Summary ----
    for r in (primary, secondary):
        verdict = "EQUIVALENT" if r["tost_95ci_in_margin"] else "NOT EQUIVALENT"
        print(f"{r['comparison']}")
        print(f"  delta (DG - AR) = {r['delta_mean_b_minus_a']:+.4f}, "
              f"95% CI [{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}], "
              f"TOST margin +/-{TOST_MARGIN}: {verdict}")
        print(f"  (classical TOST 90% CI [{r['ci_90_low']:+.4f}, "
              f"{r['ci_90_high']:+.4f}]: "
              f"{'EQUIVALENT' if r['tost_classical_90ci_equivalent'] else 'NOT EQUIVALENT'})")
        print()

    out = {
        "issue": 75,
        "description": ("Paired bootstrap CI for the AR-3B vs DiffusionGemma "
                        "9-field F1 equivalence claim."),
        "method": "paired_bootstrap",
        "metric": "per-event 9-field macro F1 (mean of 9 single-field token-F1)",
        "metric_identity_note": ("Mean of per-event 9-field F1 exactly equals the "
                                  "reported macro_field_f1_9 because all 9 fields "
                                  "are scored on all n events."),
        "n_resamples": N_BOOT,
        "seed": SEED,
        "tost_margin": TOST_MARGIN,
        "provenance": {
            "gold": "datasets/splits/test/bgl.jsonl (first N records, same order both arms)",
            "ar3b_preds": "results/issue43/ar3b_L256_preds.jsonl (key 'raw', n=300)",
            "dg_preds": "results/issue43/dg_L256_preds.jsonl (key 'raw', n=300)",
            "ar1b_preds": "results/issue_evidence_ar1b_L512_preds.jsonl (key 'raw', n=100)",
            "aggregated": "results/issue43_nine_field_f1.json",
            "parser": "models/experiments/rescore_parsers.py:parse_first_obj",
            "dict_to_record": "models/baselines/ar_wrapper.py:_dict_to_record",
            "field_f1_single": "metrics/quality_metrics.py:field_f1_single",
            "nine_fields": "metrics/quality_metrics.py:NINE_FIELD_NAMES",
        },
        "comparisons": [primary, secondary],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
