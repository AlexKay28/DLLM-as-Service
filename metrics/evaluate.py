"""
Full evaluation runner: computes all metrics from predicted and gold records.

Usage:
    python evaluate.py --pred predictions.jsonl --gold gold.jsonl --latencies latencies.txt

Expected JSONL format (one JSON object per line):
    {"timestamp_norm": "0813 09:21:14", "host": "host1", "component": "HDFS", ...}
"""

import argparse
import json
import sys
from pathlib import Path

from schema import Record, FIELD_NAMES
from quality_metrics import field_f1, schema_validity, consistency, anomaly_f1
from latency_metrics import latency_percentiles, throughput_at_sla, energy_per_event


def load_records(path: str) -> list[Record]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(Record(fields=data))
    return records


def load_latencies(path: str) -> list[float]:
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="FlowFill evaluation metrics")
    parser.add_argument("--pred", required=True, help="Predicted records (JSONL)")
    parser.add_argument("--gold", required=True, help="Gold records (JSONL)")
    parser.add_argument("--latencies", help="Per-event latencies in ms (one per line)")
    parser.add_argument("--tokens-per-event", type=int, default=128, help="Avg tokens per record")
    parser.add_argument("--sla-p99-ms", type=float, default=250.0, help="p99 SLA in ms")
    parser.add_argument("--avg-power-w", type=float, help="Avg GPU power in watts")
    parser.add_argument("--duration-s", type=float, help="Measurement duration in seconds")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    pred_records = load_records(args.pred)
    gold_records = load_records(args.gold)

    print(f"Loaded {len(pred_records)} predictions, {len(gold_records)} gold records")
    if len(pred_records) != len(gold_records):
        print("ERROR: record counts don't match")
        sys.exit(1)

    results: dict = {}

    # Quality metrics
    print("\n=== Quality Metrics ===")
    f1_results = field_f1(pred_records, gold_records)
    results["field_f1"] = f1_results
    print(f"  Field-F1 (macro):     {f1_results['macro_field_f1']:.4f}")
    for name in FIELD_NAMES:
        print(f"    {name:20s}  {f1_results[f'f1_{name}']:.4f}")

    sv = schema_validity(pred_records)
    results["schema_valid_pct"] = sv
    print(f"  Schema-valid %:       {sv:.1f}")

    cons = consistency(pred_records)
    results["consistency_pct"] = cons
    print(f"  Consistency %:        {cons:.1f}")

    anom = anomaly_f1(pred_records, gold_records)
    results["anomaly_f1"] = anom
    print(f"  Anomaly-F1:           {anom['anomaly_f1']:.4f}")
    print(f"    precision:          {anom['anomaly_precision']:.4f}")
    print(f"    recall:             {anom['anomaly_recall']:.4f}")

    # Latency metrics
    if args.latencies:
        print("\n=== Latency Metrics ===")
        latencies = load_latencies(args.latencies)
        lat = latency_percentiles(latencies)
        results["latency"] = lat
        print(f"  p50:                  {lat['p50_ms']:.1f} ms")
        print(f"  p90:                  {lat['p90_ms']:.1f} ms")
        print(f"  p99:                  {lat['p99_ms']:.1f} ms")
        print(f"  p99.9:                {lat['p99_9_ms']:.1f} ms")

        thr = throughput_at_sla(latencies, args.tokens_per_event, args.sla_p99_ms)
        results["throughput"] = {
            "max_batch_at_sla": thr.max_batch_at_sla,
            "events_per_s": thr.events_per_s,
            "tokens_per_s": thr.tokens_per_s,
            "sla_p99_ms": args.sla_p99_ms,
        }
        print(f"\n=== Throughput @ {args.sla_p99_ms}ms p99 SLA ===")
        print(f"  Max batch @SLA:       {thr.max_batch_at_sla}")
        print(f"  Events/s:             {thr.events_per_s:.1f}")
        print(f"  Tokens/s:             {thr.tokens_per_s:.0f}")

    # Energy metrics
    if args.avg_power_w and args.duration_s:
        energy = energy_per_event(args.avg_power_w, args.duration_s, len(pred_records))
        results["energy"] = {"j_per_event": energy, "avg_power_w": args.avg_power_w}
        print(f"\n=== Energy Metrics ===")
        print(f"  J/event:              {energy:.2f}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
