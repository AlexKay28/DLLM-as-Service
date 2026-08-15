#!/usr/bin/env python3
"""Issue #49 — energy measurement for MDM (LLaDA-8B) via HF wrapper."""

import json
import sys
import time
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.baselines.mdm_wrapper import MDMModel
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from metrics.schema import Record
from metrics.quality_metrics import field_f1, schema_validity, consistency
from models.experiments.power import PowerMonitor, measure_idle_power, measure_peak_mem_gb, cost_per_event

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--output", default="results/issue49_mdm.json")
    args = ap.parse_args()

    print("Measuring idle power (5s)...", flush=True)
    idle_power = measure_idle_power(device=0, duration_s=5)
    print(f"  idle power: {idle_power:.1f} W", flush=True)

    log_lines = []
    with open(str(SPLIT_DIR / "bgl.raw")) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            line = line.strip()
            if line:
                log_lines.append(line)

    gold = []
    with open(str(SPLIT_DIR / "bgl.jsonl")) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            line = line.strip()
            if line:
                gold.append(Record(fields=json.loads(line)))

    print(f"Loaded {len(log_lines)} events", flush=True)

    mdm = MDMModel(model_name=args.model, k_steps=args.k_steps, device="cuda")
    print(f"Cold-start: loading {args.model}...", flush=True)
    t_cold = time.perf_counter()
    mdm.load()
    cold_start_s = time.perf_counter() - t_cold
    print(f"  cold-start: {cold_start_s:.1f}s", flush=True)

    import torch
    torch.cuda.reset_peak_memory_stats()

    pm = PowerMonitor(device=0, interval=0.05)
    pm.start()

    print(f"Streaming {len(log_lines)} events (m=1, K={args.k_steps})...", flush=True)
    result = mdm.extract(log_lines, source="bgl", show_progress=False,
                         k_steps=args.k_steps)
    latencies = result.latencies_ms
    records = result.records
    print(f"  done. {len(records)} records, p50={statistics.median(latencies):.1f}ms",
          flush=True)

    pm.stop()
    wall_s = pm.duration_s()
    peak_mem = measure_peak_mem_gb(0)

    f1_res = field_f1(records, gold[:len(records)])
    quality = {
        "field_f1": round(f1_res["macro_field_f1"], 4),
        "schema_valid_pct": round(schema_validity(records), 2),
        "consistency_pct": round(consistency(records), 2),
    }

    energy = pm.energy_j(idle_power)
    j_per_event = energy / len(records) if records else 0
    costs = cost_per_event(j_per_event)

    result = {
        "model": args.model,
        "n_events": len(records),
        "k_steps": args.k_steps,
        "cold_start_s": round(cold_start_s, 2),
        "power": pm.summary(idle_power_w=idle_power, n_events=len(records)),
        "peak_mem_gb": peak_mem,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p99": round(sorted(latencies)[int(len(latencies) * 0.99)], 1),
            "mean": round(statistics.fmean(latencies), 1),
        },
        "quality": quality,
        "energy_per_event_j": round(j_per_event, 4),
        "cost": costs,
        "wall_time_s": round(wall_s, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    print(f"\n=== Results ===")
    print(f"  cold-start: {cold_start_s:.1f}s")
    print(f"  idle power: {idle_power:.1f} W")
    print(f"  avg power: {pm.avg_power_watts():.1f} W")
    print(f"  peak power: {pm.peak_power_watts():.1f} W")
    print(f"  energy: {energy:.1f} J total, {j_per_event:.4f} J/event")
    print(f"  peak mem: {peak_mem} GB")
    print(f"  p50 latency: {result['latency_ms']['p50']:.1f} ms")
    print(f"  F1: {quality['field_f1']:.3f}")
    print(f"  $/event: {costs['total_per_event']:.6f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
