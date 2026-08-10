"""
Validate throughput @ SLA claims from FlowFill paper Table 4 (Issue #5).

Packed mode (m>1, batched). Sweeps batch sizes [1..128] to find the max
batch that satisfies p99 <= 250ms SLA, then reports events/s and tokens/s.

Usage (GPU required):
  python models/experiments/run_throughput_table4.py --model Qwen/Qwen2.5-1.5B-Instruct
  python models/experiments/run_throughput_table4.py \
    --model Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct

  # Dry-run (no GPU)
  python models/experiments/run_throughput_table4.py --mock

Outputs:
  results/throughput_table4.json
  results/throughput_table4.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.latency_metrics import latency_percentiles

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
RESULTS_DIR = _REPO_ROOT / "results"

PAPER_TABLE4 = {
    "AR-1.5B":            {"max_batch": 24, "events_s": 640,  "tokens_s": 82_000},
    "AR-1.5B + grammar":  {"max_batch": 20, "events_s": 520,  "tokens_s": 67_000},
    "FlowFill-MDM (K=8)": {"max_batch": 96, "events_s": 4350, "tokens_s": 557_000},
    "AR-7B":              {"max_batch": 6,  "events_s": 150,  "tokens_s": 19_000},
}

# Batch sizes to sweep (paper uses up to 128)
BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
SLA_P99_MS = 250.0
TOKENS_PER_EVENT = 128  # paper's assumption


def load_log_lines(path, n=None):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def sweep_batch_sizes(model, log_lines, source, model_name,
                      batch_sizes=None, events_per_batch=500):
    """Sweep batch sizes and measure per-event p99 at each.

    Args:
        events_per_batch: number of events to run at each batch size
            (more = more stable p99, but slower).
    """
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES

    results = []
    n_available = len(log_lines)

    for batch_size in batch_sizes:
        # Use a subset of events for this batch size test
        n_test = min(events_per_batch, n_available)
        test_lines = log_lines[:n_test]

        print(f"\n  Batch size {batch_size} (n={n_test})...", end=" ", flush=True)

        # Set the model's batch size
        model.batch_size = batch_size

        t0 = time.perf_counter()
        result = model.extract(test_lines, source=source, show_progress=False)
        elapsed = time.perf_counter() - t0

        latencies = result.latencies_ms
        pct = latency_percentiles(latencies)
        p99 = pct.get("p99_ms", 0)
        p50 = pct.get("p50_ms", 0)

        events_s = batch_size * (1000.0 / p50) if p50 > 0 else 0
        tokens_s = events_s * TOKENS_PER_EVENT
        meets_sla = p99 <= SLA_P99_MS

        print(f"p50={p50:.1f}ms p99={p99:.1f}ms ev/s={events_s:.0f} "
              f"{'SLA-OK' if meets_sla else 'SLA-FAIL'}")

        results.append({
            "batch_size": batch_size,
            "p50_ms": round(p50, 2),
            "p99_ms": round(p99, 2),
            "p999_ms": round(pct.get("p99_9_ms", 0), 2),
            "events_s": round(events_s, 1),
            "tokens_s": round(tokens_s, 0),
            "meets_sla": meets_sla,
            "n_tested": n_test,
        })

    return results


def find_max_batch_at_sla(sweep_results):
    """Find the largest batch size where p99 <= SLA."""
    max_batch = 1
    best_events_s = 0
    best_tokens_s = 0
    for r in sweep_results:
        if r["meets_sla"] and r["events_s"] > best_events_s:
            max_batch = r["batch_size"]
            best_events_s = r["events_s"]
            best_tokens_s = r["tokens_s"]
    return max_batch, best_events_s, best_tokens_s


def write_report(all_results, path):
    lines = [
        "# Throughput @ SLA Claims — Issue #5: Table 4",
        "",
        f"**Data:** BGL (LOGHUB), packed mode (m>1)",
        f"**SLA:** p99 <= {SLA_P99_MS:.0f}ms",
        f"**Date:** 2026-08-11",
        "",
        "## Results vs Paper Table 4",
        "",
        "| Model | Max Batch @SLA | events/s | tokens/s |",
        "|-------|---------------|----------|----------|",
    ]
    for r in all_results:
        lines.append(f"| {r['model']} (ours) | {r['max_batch']} | "
                     f"{r['events_s']:.0f} | {r['tokens_s']:.0f} |")
    lines.append("| | | | |")
    for name, p in PAPER_TABLE4.items():
        lines.append(f"| {name} (paper) | {p['max_batch']} | "
                     f"{p['events_s']} | {p['tokens_s']:,} |")

    lines.extend([
        "",
        "## Batch Size Sweep Detail",
        "",
    ])
    for r in all_results:
        lines.append(f"### {r['model']}")
        lines.append("")
        lines.append("| Batch | p50 (ms) | p99 (ms) | events/s | SLA |")
        lines.append("|-------|---------|---------|----------|-----|")
        for s in r["sweep"]:
            lines.append(f"| {s['batch_size']} | {s['p50_ms']:.1f} | "
                         f"{s['p99_ms']:.1f} | {s['events_s']:.0f} | "
                         f"{'OK' if s['meets_sla'] else 'FAIL'} |")
        lines.append("")

    lines.extend([
        "## Allegations to verify",
        "",
        "1. FlowFill 6.8x throughput of AR-1.5B (4350 vs 640 ev/s)",
        "2. FlowFill packs 4x the batch (96 vs 24)",
        "3. Grammar reduces AR throughput ~19%",
        "4. AR-7B collapses to 150 ev/s (batch=6)",
        "5. tokens/s ratio matches events/s ratio (implies ~128 tok/event)",
        "6. iso-SLA framing is the only axis diffusion wins",
        "",
        "**Note:** FlowFill column blocked on weights. AR substitutes use Qwen2.5.",
        "",
        "---",
        "*Generated by `models/experiments/run_throughput_table4.py`*",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct"])
    parser.add_argument("--source", default="bgl")
    parser.add_argument("--n", type=int, default=2000,
                        help="Events per batch size (default 2000)")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--sla-ms", type=float, default=SLA_P99_MS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.source} test data...")
    log_lines = load_log_lines(str(SPLIT_DIR / f"{args.source}.raw"), n=args.n * 4)
    print(f"  {len(log_lines):,} log lines available")

    all_results = []
    for model_id in args.model:
        short_name = model_id.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Model: {short_name}")
        print(f"{'='*60}")

        if args.mock:
            from models.baselines.ar_wrapper import MockARModel
            model = MockARModel(model_name=short_name)
        else:
            from models.baselines.ar_wrapper import ARModel
            model = ARModel(model_name=model_id, device=args.device,
                           dtype=args.dtype, max_new_tokens=256, batch_size=1)

        sweep = sweep_batch_sizes(model, log_lines, args.source, short_name,
                                   events_per_batch=args.n)
        max_batch, events_s, tokens_s = find_max_batch_at_sla(sweep)

        all_results.append({
            "model": short_name,
            "max_batch": max_batch,
            "events_s": events_s,
            "tokens_s": tokens_s,
            "sweep": sweep,
        })

    output = {"source": args.source, "sla_p99_ms": args.sla_ms,
              "tokens_per_event": TOKENS_PER_EVENT,
              "models": all_results, "paper_table4": PAPER_TABLE4}
    json_path = RESULTS_DIR / "throughput_table4.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    write_report(all_results, RESULTS_DIR / "throughput_table4.md")
    print("Done!")


if __name__ == "__main__":
    main()
