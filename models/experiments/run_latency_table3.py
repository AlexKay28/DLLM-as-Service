"""
Validate latency & energy claims from FlowFill paper Table 3 (Issue #4).

Streaming mode (m=1, single event per context). Measures p50/p99/p99.9
latency and J/event for AR model substitutes.

Usage (GPU required):
  python models/experiments/run_latency_table3.py --model Qwen/Qwen2.5-1.5B-Instruct --n 10000
  python models/experiments/run_latency_table3.py \
    --model Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct --n 10000

  # Dry-run (no GPU)
  python models/experiments/run_latency_table3.py --mock --n 500

Outputs:
  results/latency_table3.json
  results/latency_table3.md
"""

import argparse
import json
import sys
import time
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, field

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.latency_metrics import latency_percentiles, energy_per_event

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
RESULTS_DIR = _REPO_ROOT / "results"

PAPER_TABLE3 = {
    "AR-1.5B":            {"steps": 128, "p50": 168, "p99": 240, "p999": 340, "j": 3.9},
    "AR-1.5B + grammar":  {"steps": 128, "p50": 205, "p99": 292, "p999": 410, "j": 4.8},
    "FlowFill-MDM (K=8)": {"steps": 8,   "p50": 23,  "p99": 31,  "p999": 36,  "j": 0.7},
    "AR-7B":              {"steps": 128, "p50": 690, "p99": 910, "p999": 1240,"j": 22.5},
}


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


class PowerMonitor:
    """Polls nvidia-smi for GPU power draw in a background thread.

    Usage:
        pm = PowerMonitor(device=0)
        pm.start()
        # ... run inference ...
        pm.stop()
        avg_power = pm.avg_power_watts()  # board-level, including idle
    """

    def __init__(self, device=0, interval=0.5):
        self.device = device
        self.interval = interval
        self._readings = []
        self._thread = None
        self._stop = False

    def start(self):
        self._stop = False
        self._readings = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

    def _poll(self):
        import time
        while not self._stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     f"--id={self.device}",
                     "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    timeout=2,
                ).decode().strip()
                if out:
                    self._readings.append(float(out.split("\n")[0]))
            except (subprocess.SubprocessError, ValueError, FileNotFoundError):
                pass
            time.sleep(self.interval)

    def avg_power_watts(self):
        if not self._readings:
            return 0.0
        return sum(self._readings) / len(self._readings)

    def has_readings(self):
        return len(self._readings) > 0


def measure_idle_power(device=0, duration_s=5):
    """Measure idle GPU power for baseline subtraction."""
    pm = PowerMonitor(device=device, interval=0.5)
    pm.start()
    time.sleep(duration_s)
    pm.stop()
    return pm.avg_power_watts()


def run_model(model, log_lines, source, model_name, device=0,
              measure_energy=True):
    """Run model in streaming mode (m=1) and measure latency + energy."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} (streaming mode m=1)")
    print(f"{'='*60}")

    # Measure idle power baseline
    idle_power = 0.0
    if measure_energy:
        print("  Measuring idle power (5s)...")
        idle_power = measure_idle_power(device=device, duration_s=5)
        print(f"  Idle power: {idle_power:.1f}W")

    # Run inference with power monitoring
    pm = PowerMonitor(device=device, interval=0.5)
    if measure_energy:
        pm.start()

    t0 = time.perf_counter()
    result = model.extract(log_lines, source=source, show_progress=True)
    duration_s = time.perf_counter() - t0

    if measure_energy:
        pm.stop()

    latencies = result.latencies_ms
    pct = latency_percentiles(latencies)

    # Energy
    j_per_event = 0.0
    if measure_energy and pm.has_readings():
        load_power = pm.avg_power_watts()
        net_power = max(load_power - idle_power, 0.0)
        j_per_event = energy_per_event(net_power, duration_s, len(log_lines))
        print(f"  Load power: {load_power:.1f}W, net: {net_power:.1f}W")
        print(f"  Energy: {j_per_event:.2f} J/event")

    # Token stats
    avg_tokens = (sum(result.token_counts) / len(result.token_counts)
                  if result.token_counts else 0)

    metrics = {
        "model": model_name,
        "n_events": len(log_lines),
        "steps": int(avg_tokens),  # AR steps = output tokens
        "p50_ms": round(pct.get("p50_ms", 0), 2),
        "p99_ms": round(pct.get("p99_ms", 0), 2),
        "p999_ms": round(pct.get("p99_9_ms", 0), 2),
        "j_per_event": round(j_per_event, 2),
        "avg_tokens_per_event": round(avg_tokens, 1),
        "total_time_s": round(duration_s, 2),
        "idle_power_w": round(idle_power, 1) if measure_energy else 0,
    }

    print(f"\n  Results:")
    print(f"    p50/p99/p99.9: {metrics['p50_ms']:.1f} / "
          f"{metrics['p99_ms']:.1f} / {metrics['p999_ms']:.1f} ms")
    print(f"    Steps (tokens): {metrics['steps']}")
    print(f"    Energy: {metrics['j_per_event']:.2f} J/event")

    return metrics


def write_report(results, path):
    lines = [
        "# Latency & Energy Claims — Issue #4: Table 3",
        "",
        f"**Data:** BGL (LOGHUB), streaming mode (m=1)",
        f"**Date:** 2026-08-11",
        "",
        "## Results vs Paper Table 3",
        "",
        "| Model | Steps | p50 (ms) | p99 (ms) | p99.9 (ms) | J/event |",
        "|-------|-------|---------|---------|-----------|---------|",
    ]
    for r in results:
        lines.append(f"| {r['model']} (ours) | {r['steps']} | {r['p50_ms']:.0f} | "
                     f"{r['p99_ms']:.0f} | {r['p999_ms']:.0f} | {r['j_per_event']:.1f} |")
    lines.append("| | | | | | |")
    for name, p in PAPER_TABLE3.items():
        lines.append(f"| {name} (paper) | {p['steps']} | {p['p50']} | "
                     f"{p['p99']} | {p['p999']} | {p['j']} |")

    lines.extend([
        "",
        "## Allegations to verify",
        "",
        "1. FlowFill 7.3x faster than AR-1.5B at p50 (23 vs 168ms)",
        "2. Latency content-independent: p99.9/p50 ratio tight for FlowFill",
        "3. Grammar constraints make AR slower (168 -> 205ms)",
        "4. Energy scales with latency (0.7 vs 3.9 J/event)",
        "5. AR = 128 steps, FlowFill = 8 steps",
        "6. p99.9 SLA fit: FlowFill 36ms fits; AR-7B 1240ms doesn't",
        "",
        "**Note:** FlowFill column blocked on weights. AR substitutes use Qwen2.5.",
        "",
        "---",
        "*Generated by `models/experiments/run_latency_table3.py`*",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct"])
    parser.add_argument("--source", default="bgl")
    parser.add_argument("--n", type=int, default=10000,
                        help="Number of events (need >=10k for stable p99.9)")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID for power")
    parser.add_argument("--no-energy", action="store_true", help="Skip energy measurement")
    parser.add_argument("--mdm", action="store_true",
                        help="Use MDM model instead of AR (for FlowFill substitute)")
    parser.add_argument("--k-steps", type=int, default=8,
                        help="MDM denoising steps (default 8)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.source} test data...")
    log_lines = load_log_lines(str(SPLIT_DIR / f"{args.source}.raw"), n=args.n)
    print(f"  {len(log_lines):,} log lines")

    all_results = []
    for model_id in args.model:
        short_name = model_id.split("/")[-1]
        if args.mock and args.mdm:
            from models.baselines.mdm_wrapper import MockMDMModel
            model = MockMDMModel(model_name=short_name, k_steps=args.k_steps)
        elif args.mock:
            from models.baselines.ar_wrapper import MockARModel
            model = MockARModel(model_name=short_name)
        elif args.mdm:
            from models.baselines.mdm_wrapper import MDMModel
            model = MDMModel(model_name=model_id, k_steps=args.k_steps,
                           device=args.device, dtype=args.dtype)
        else:
            from models.baselines.ar_wrapper import ARModel
            model = ARModel(model_name=model_id, device=args.device,
                           dtype=args.dtype, max_new_tokens=256, batch_size=1)

        # Update steps label for MDM (K steps, not 128 tokens)
        if args.mdm or (args.mock and args.mdm):
            short_name += f" (K={args.k_steps})"

        metrics = run_model(model, log_lines, args.source, short_name,
                           device=args.gpu_id, measure_energy=not args.no_energy)
        all_results.append(metrics)

    output = {"source": args.source, "n_events": len(log_lines),
              "models": all_results, "paper_table3": PAPER_TABLE3}
    json_path = RESULTS_DIR / "latency_table3.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    write_report(all_results, RESULTS_DIR / "latency_table3.md")
    print("Done!")


if __name__ == "__main__":
    main()
