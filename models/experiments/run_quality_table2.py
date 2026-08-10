"""
Validate quality claims from FlowFill paper Table 2 (Issue #3).

Runs AR model substitutes (Qwen2.5-1.5B, Qwen2.5-7B) on real BGL test data
and computes the 4 quality metrics: Field-F1, Schema-validity, Consistency,
Anomaly-F1. Compares to the paper's Table 2 numbers.

Usage (GPU required):
  # Full run with real AR model
  python models/experiments/run_quality_table2.py --model Qwen/Qwen2.5-1.5B-Instruct

  # With 7B model too
  python models/experiments/run_quality_table2.py \
    --model Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct

  # Dry-run without GPU (uses MockARModel to verify pipeline)
  python models/experiments/run_quality_table2.py --mock --n 500

Outputs:
  results/quality_table2.json — metrics for each model
  results/quality_table2.md   — comparison to paper Table 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, FIELD_BY_NAME, FieldCategory
from metrics.quality_metrics import field_f1, schema_validity, consistency, anomaly_f1
from metrics.latency_metrics import latency_percentiles, energy_per_event

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
RESULTS_DIR = _REPO_ROOT / "results"

# Paper Table 2 values (all 4 models, for comparison)
PAPER_TABLE2 = {
    "AR-1.5B": {
        "field_f1": 0.913, "schema_valid_pct": 96.1,
        "consistency_pct": 98.9, "anomaly_f1": 0.871,
        "p50_ms": 168, "p99_ms": 240, "p999_ms": 340, "j_per_event": 3.9,
        "steps": 128,
    },
    "AR-1.5B + grammar": {
        "field_f1": 0.916, "schema_valid_pct": 100.0,
        "consistency_pct": 99.0, "anomaly_f1": 0.874,
        "p50_ms": 205, "p99_ms": 292, "p999_ms": 410, "j_per_event": 4.8,
        "steps": 128,
    },
    "FlowFill-MDM (K=8)": {
        "field_f1": 0.907, "schema_valid_pct": 99.4,
        "consistency_pct": 97.8, "anomaly_f1": 0.865,
        "p50_ms": 23, "p99_ms": 31, "p999_ms": 36, "j_per_event": 0.7,
        "steps": 8,
    },
    "AR-7B": {
        "field_f1": 0.945, "schema_valid_pct": 97.4,
        "consistency_pct": 99.4, "anomaly_f1": 0.912,
        "p50_ms": 690, "p99_ms": 910, "p999_ms": 1240, "j_per_event": 22.5,
        "steps": 128,
    },
}


def load_log_lines(path: str, n: int = None) -> list[str]:
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_records(path: str, n: int = None) -> list[Record]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                records.append(Record(fields=json.loads(line)))
    return records


def evaluate_model(records, gold_records, latencies_ms, model_name):
    """Compute all Table 2 metrics for a model's predictions."""
    f1_res = field_f1(records, gold_records)
    sv = schema_validity(records)
    cons = consistency(records)
    anom = anomaly_f1(records, gold_records)
    lat = latency_percentiles(latencies_ms) if latencies_ms else {}

    return {
        "model": model_name,
        "field_f1": round(f1_res["macro_field_f1"], 4),
        "per_field_f1": {k: round(v, 4) for k, v in f1_res.items()
                         if k.startswith("f1_")},
        "schema_valid_pct": round(sv, 2),
        "consistency_pct": round(cons, 2),
        "anomaly_f1": round(anom["anomaly_f1"], 4),
        "p50_ms": round(lat.get("p50_ms", 0), 2),
        "p99_ms": round(lat.get("p99_ms", 0), 2),
        "p999_ms": round(lat.get("p99_9_ms", 0), 2),
        "n_events": len(records),
    }


def run_model(model, log_lines, gold_records, source, model_name):
    """Run a single model on the test data and evaluate."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    t0 = time.time()
    result = model.extract(log_lines, source=source, show_progress=True)
    elapsed = time.time() - t0
    print(f"  Extracted {len(result.records)} records in {elapsed:.1f}s")
    print(f"  Avg tokens/event: "
          f"{sum(result.token_counts)/len(result.token_counts):.0f}" if result.token_counts else "")
    print(f"  Latency p50/p99: "
          f"{result.latencies_ms[len(result.latencies_ms)//2]:.1f}ms / "
          f"{sorted(result.latencies_ms)[int(len(result.latencies_ms)*0.99)]:.1f}ms"
          if result.latencies_ms else "")

    metrics = evaluate_model(
        result.records, gold_records, result.latencies_ms, model_name)
    metrics["total_time_s"] = round(elapsed, 2)
    metrics["avg_tokens_per_event"] = round(
        sum(result.token_counts) / len(result.token_counts), 1) if result.token_counts else 0

    print(f"\n  Results:")
    print(f"    Field-F1:       {metrics['field_f1']:.4f}")
    print(f"    Schema-valid:   {metrics['schema_valid_pct']:.1f}%")
    print(f"    Consistency:    {metrics['consistency_pct']:.1f}%")
    print(f"    Anomaly-F1:     {metrics['anomaly_f1']:.4f}")
    print(f"    p50/p99/p99.9:  {metrics['p50_ms']:.1f} / "
          f"{metrics['p99_ms']:.1f} / {metrics['p999_ms']:.1f} ms")

    return metrics


def write_report(results, paper_data, path):
    """Write the Table 2 comparison report."""
    lines = [
        "# Quality Claims Validation — Issue #3: Table 2",
        "",
        f"**Data:** BGL (LOGHUB) — {results[0]['n_events']:,} test events" if results else "",
        f"**Date:** 2026-08-11",
        "",
        "---",
        "",
        "## Results vs Paper Table 2",
        "",
        "| Model | Field-F1 | Schema-Valid | Consistency | Anomaly-F1 | p50 (ms) | p99 (ms) |",
        "|-------|---------|-------------|-------------|-----------|---------|---------|",
    ]

    # Our results
    for r in results:
        name = r["model"]
        lines.append(
            f"| {name} (ours) | {r['field_f1']:.3f} | {r['schema_valid_pct']:.1f}% | "
            f"{r['consistency_pct']:.1f}% | {r['anomaly_f1']:.3f} | "
            f"{r['p50_ms']:.0f} | {r['p99_ms']:.0f} |")

    lines.append("| | | | | | | |")

    # Paper values
    for name, p in paper_data.items():
        lines.append(
            f"| {name} (paper) | {p['field_f1']:.3f} | {p['schema_valid_pct']:.1f}% | "
            f"{p['consistency_pct']:.1f}% | {p['anomaly_f1']:.3f} | "
            f"{p['p50_ms']} | {p['p99_ms']} |")

    lines.extend([
        "",
        "## Allegation Verification",
        "",
    ])

    # Compare our AR-1.5B substitute to paper's AR-1.5B
    our_ar15 = next((r for r in results if "1.5B" in r["model"]), None)
    paper_ar15 = paper_data.get("AR-1.5B", {})

    if our_ar15 and paper_ar15:
        ff1_gap = our_ar15["field_f1"] - paper_ar15["field_f1"]
        lines.append(f"### 1. FlowFill trails AR-1.5B by 0.006 Field-F1")
        lines.append(f"- Our AR-1.5B substitute: {our_ar15['field_f1']:.3f}")
        lines.append(f"- Paper AR-1.5B: {paper_ar15['field_f1']:.3f}")
        lines.append(f"- Gap: {ff1_gap:+.3f}")
        lines.append("")

    # Grammar constraint effect
    lines.append("### 2. Grammar constraints buy validity, not quality")
    lines.append(f"- Paper: AR-1.5B validity 96.1% → +grammar 100.0%, F1 0.913 → 0.916")
    lines.append(f"- To verify: run with --grammar flag (needs XGrammar)")
    lines.append("")

    # Consistency
    our_ar7 = next((r for r in results if "7B" in r["model"]), None)
    paper_ar7 = paper_data.get("AR-7B", {})
    if our_ar15 and paper_ar15:
        cons_gap = our_ar15["consistency_pct"] - paper_ar15["consistency_pct"]
        lines.append(f"### 3. FlowFill loses ~1.1pp consistency vs AR")
        lines.append(f"- Our AR substitute consistency: {our_ar15['consistency_pct']:.1f}%")
        lines.append(f"- Paper AR-1.5B: {paper_ar15['consistency_pct']:.1f}%")
        lines.append(f"- Paper FlowFill: {paper_data.get('FlowFill-MDM (K=8)', {}).get('consistency_pct', '?')}%")
        lines.append("")

    if our_ar7 and paper_ar7:
        lines.append(f"### 4. AR-7B is the Field-F1 ceiling but lower validity")
        lines.append(f"- Our AR-7B F1: {our_ar7['field_f1']:.3f}, validity: {our_ar7['schema_valid_pct']:.1f}%")
        lines.append(f"- Paper AR-7B F1: {paper_ar7['field_f1']:.3f}, validity: {paper_ar7['schema_valid_pct']:.1f}%")
        lines.append("")

    # Anomaly-F1 ordering
    lines.append("### 5. Anomaly-F1 ordering: AR-7B > AR-1.5B > FlowFill")
    lines.append(f"- Paper: AR-7B (0.912) > AR-1.5B (0.871) > FlowFill (0.865)")
    if our_ar15:
        lines.append(f"- Our AR-1.5B substitute: {our_ar15['anomaly_f1']:.3f}")
    if our_ar7:
        lines.append(f"- Our AR-7B substitute: {our_ar7['anomaly_f1']:.3f}")
    lines.append("")

    lines.extend([
        "## Caveats",
        "",
        "1. **Model substitution**: we use open Qwen2.5 models as AR substitutes, "
        "not the paper's exact AR-1.5B/AR-7B. F1 differences reflect model + "
        "prompt differences, not just the diffusion-vs-AR question.",
        "2. **Gold labels**: semantic fields (event_type, affected_entity, "
        "remediation_hint) are self-generated placeholders. AR models that "
        "produce real semantic labels will be scored against weak gold — "
        "their semantic F1 may appear artificially low.",
        "3. **FlowFill column**: blocked on weights. Paper values shown for reference.",
        "4. **Single corpus**: only BGL. Paper averages 4 corpora.",
        "",
        "---",
        f"*Generated by `models/experiments/run_quality_table2.py`*",
    ])

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct"],
                        help="HuggingFace model ID(s) to use as AR substitutes")
    parser.add_argument("--source", default="bgl",
                        choices=["hdfs", "bgl", "thunderbird", "spirit"])
    parser.add_argument("--n", type=int, default=None,
                        help="Number of test events (default: all 200k)")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockARModel (no GPU needed, for testing)")
    parser.add_argument("--device", default="cuda", help="Device: cuda, cpu, cuda:0")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (1=streaming mode for Table 3; >1 for Table 4)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--grammar", action="store_true",
                        help="Use grammar-constrained decoding (XGrammar)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    print(f"Loading {args.source} test data...")
    log_lines = load_log_lines(str(SPLIT_DIR / f"{args.source}.raw"), n=args.n)
    gold_records = load_records(str(SPLIT_DIR / f"{args.source}.jsonl"), n=args.n)
    print(f"  {len(log_lines):,} log lines, {len(gold_records):,} gold records")

    all_results = []

    for model_id in args.model:
        # Short name for display
        short_name = model_id.split("/")[-1]
        if args.grammar:
            short_name += " + grammar"

        if args.mock:
            from models.baselines.ar_wrapper import MockARModel
            model = MockARModel(model_name=short_name)
        else:
            from models.baselines.ar_wrapper import ARModel
            model = ARModel(
                model_name=model_id,
                device=args.device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                grammar_constrained=args.grammar,
                batch_size=args.batch_size,
            )

        metrics = run_model(model, log_lines, gold_records, args.source, short_name)
        all_results.append(metrics)

    # Save JSON
    output = {
        "source": args.source,
        "n_events": all_results[0]["n_events"] if all_results else 0,
        "models": all_results,
        "paper_table2": PAPER_TABLE2,
    }
    json_path = RESULTS_DIR / "quality_table2.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Write report
    write_report(all_results, PAPER_TABLE2, RESULTS_DIR / "quality_table2.md")

    print("\nDone!")


if __name__ == "__main__":
    main()
