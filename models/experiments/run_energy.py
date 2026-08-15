#!/usr/bin/env python3
"""Issue #49 — measure energy per event for AR (vLLM) and DG (vLLM).

Measures: idle power, avg/peak power during inference, energy per event,
peak GPU memory, cold-start time, and $/event estimate.

Usage:
    python models/experiments/run_energy.py --model Qwen/Qwen2.5-3B-Instruct --n 300
    python models/experiments/run_energy.py --model google/diffusiongemma-26B-A4B-it --denoising-steps 8 --n 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experiments.power import (
    PowerMonitor, measure_idle_power, measure_peak_mem_gb, cost_per_event,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--denoising-steps", type=int, default=None)
    ap.add_argument("--source", default="bgl")
    ap.add_argument("--output", default="results/issue49_energy.json")
    ap.add_argument("--gpu-rate", type=float, default=2.50,
                    help="GPU rental rate $/hour")
    ap.add_argument("--electricity-rate", type=float, default=0.12,
                    help="electricity rate $/kWh")
    args = ap.parse_args()

    print("Measuring idle power (5s)...", flush=True)
    idle_power = measure_idle_power(device=0, duration_s=5)
    print(f"  idle power: {idle_power:.1f} W", flush=True)

    from vllm import LLM, SamplingParams
    from metrics.schema import Record
    from metrics.quality_metrics import field_f1, schema_validity, consistency
    from models.baselines.ar_model import build_chat_messages, parse_json_output
    from models.baselines.ar_wrapper import _dict_to_record

    SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"

    def load_lines(path, n):
        lines = []
        with open(path) as f:
            for i, line in enumerate(f):
                if n and i >= n:
                    break
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def load_gold(path, n):
        records = []
        with open(path) as f:
            for i, line in enumerate(f):
                if n and i >= n:
                    break
                line = line.strip()
                if line:
                    records.append(Record(fields=json.loads(line)))
        return records

    log_lines = load_lines(str(SPLIT_DIR / f"{args.source}.raw"), args.n)
    gold = load_gold(str(SPLIT_DIR / f"{args.source}.jsonl"), args.n)
    print(f"Loaded {len(log_lines)} events", flush=True)

    llm_kwargs = dict(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=4096,
        enable_prefix_caching=True,
    )
    if args.denoising_steps is not None:
        llm_kwargs["diffusion_config"] = {
            "canvas_length": 256,
            "max_denoising_steps": args.denoising_steps}

    print(f"\nCold-start: loading {args.model}...", flush=True)
    t_cold = time.perf_counter()
    llm = LLM(**llm_kwargs)
    cold_start_s = time.perf_counter() - t_cold
    print(f"  cold-start: {cold_start_s:.1f}s", flush=True)

    tok = llm.get_tokenizer()
    prompts = []
    for line in log_lines:
        messages = build_chat_messages(line, source=args.source, few_shot=True)
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(
        max_tokens=args.max_new_tokens, min_tokens=args.min_tokens,
        temperature=0.0)

    import torch
    torch.cuda.reset_peak_memory_stats()

    pm = PowerMonitor(device=0, interval=0.05)
    pm.start()

    latencies = []
    records = []
    out_tokens = []
    print(f"Streaming {len(prompts)} events (m=1)...", flush=True)
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        text = outs[0].outputs[0].text
        out_tokens.append(len(outs[0].outputs[0].token_ids))
        records.append(_dict_to_record(parse_json_output(text)))
        if (i + 1) % 100 == 0:
            import statistics
            print(f"  {i+1}/{len(prompts)}  p50={statistics.median(latencies):.1f}ms",
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
    costs = cost_per_event(j_per_event, args.gpu_rate, args.electricity_rate)

    import statistics as st
    result = {
        "model": args.model,
        "n_events": len(records),
        "max_new_tokens": args.max_new_tokens,
        "min_tokens": args.min_tokens,
        "cold_start_s": round(cold_start_s, 2),
        "power": pm.summary(idle_power_w=idle_power, n_events=len(records)),
        "peak_mem_gb": peak_mem,
        "latency_ms": {
            "p50": round(st.median(latencies), 1),
            "p99": round(sorted(latencies)[int(len(latencies) * 0.99)], 1),
            "mean": round(st.fmean(latencies), 1),
        },
        "mean_output_tokens": round(st.fmean(out_tokens), 1),
        "quality": quality,
        "energy_per_event_j": round(j_per_event, 4),
        "cost": costs,
        "wall_time_s": round(wall_s, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if args.denoising_steps is not None:
        result["max_denoising_steps"] = args.denoising_steps

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

    del llm
    import gc, torch as _torch
    gc.collect()
    _torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
