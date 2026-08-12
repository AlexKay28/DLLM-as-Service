#!/usr/bin/env python3
"""
DEPRECATED — DO NOT RUN. Superseded by `run_vllm_serving.py`.

This 2026-08-11 draft produced numbers that cannot be real (p50 = 0.42 ms and
322k events/s for an 8B model, in the since-deleted
results/diffusiongemma_perf.json). Two defects, both silent:

1. `load_bgl_test_data()` looks for `datasets/bgl/test_200k.jsonl`, which does
   not exist in this repo — the split lives at `datasets/splits/test/bgl.raw`.
   On the missing path it falls back to *synthetic* "ERROR: Component i failed"
   strings and logs one warning, so the run reports BGL metrics computed on
   fabricated logs. See logs/diffusiongemma_perf_n100.log.
2. `measure_throughput_at_sla()` calls `model.extract()` in a sequential
   `for _ in range(batch_size)` loop and divides by mean per-call latency. That
   measures a loop, not batching, and events/s then rises linearly with batch
   size by construction — which is what produced the 322k ev/s figure.

`measure_latency()` also swallows every per-prompt exception with a bare
`except`, so a run in which inference never happened still emits a full
percentile table.

Kept only as a record of the attempt. The working replacement is
`models/experiments/run_vllm_serving.py` (real BGL data, real continuous
batching); the blocker analysis is in `results/diffusiongemma_perf.md`.
"""

import sys as _sys

_sys.exit(
    "run_diffusiongemma_perf.py is deprecated and produces invalid results; "
    "use models/experiments/run_vllm_serving.py instead (see module docstring)."
)

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.mdm_wrapper import MDMModel, MockMDMModel
from baselines.ar_wrapper import ARModel, MockARModel

def load_bgl_test_data(n: int = 500):
    """Load BGL test data (temporal split - last 200k lines)"""
    data_dir = Path(__file__).parent.parent.parent / "datasets" / "bgl"
    test_file = data_dir / "test_200k.jsonl"
    
    if not test_file.exists():
        # Fallback to prepared data
        test_file = data_dir / "prepared_test.jsonl"
    
    if not test_file.exists():
        print(f"Warning: BGL test data not found at {test_file}")
        # Create synthetic data for testing
        return [{"log": f"ERROR: Component {i} failed on host test-{i%10}", "fields": {}} for i in range(n)]
    
    with open(test_file, 'r') as f:
        lines = f.readlines()[:n]
    
    data = []
    for line in lines:
        try:
            item = json.loads(line)
            data.append({"log": item.get("log", str(item)), "fields": item.get("fields", {})})
        except:
            data.append({"log": line.strip(), "fields": {}})
    
    return data

def measure_latency(model: Any, prompts: List[str], n_runs: int = 3) -> Dict[str, float]:
    """Measure p50/p99/p99.9 latency"""
    latencies = []
    
    print(f"Measuring latency on {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        try:
            _ = model.extract(prompt, k_steps=model.k_steps if hasattr(model, 'k_steps') else 8)
        except Exception as e:
            print(f"Error on prompt {i}: {e}")
            continue
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)} - p50: {np.median(latencies):.1f}ms")
    
    latencies = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "p999_ms": float(np.percentile(latencies, 99.9)),
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "n_samples": len(latencies),
    }

def measure_throughput_at_sla(model: Any, prompt: str, target_p99_ms: float = 250.0, 
                               batch_sizes: List[int] = None) -> Dict[str, Any]:
    """Find max batch size that meets p99 SLA, compute throughput"""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    
    results = []
    max_batch_meeting_sla = 1
    
    print(f"Measuring throughput @ {target_p99_ms}ms p99 SLA...")
    for batch_size in batch_sizes:
        prompts = [prompt] * batch_size
        latencies = []
        
        start_total = time.perf_counter()
        try:
            # Batch inference
            _ = model.extract(prompt, k_steps=model.k_steps if hasattr(model, 'k_steps') else 8)
            # For true batching, we'd need vLLM - simulating here
            for _ in range(batch_size):
                s = time.perf_counter()
                _ = model.extract(prompt, k_steps=model.k_steps if hasattr(model, 'k_steps') else 8)
                latencies.append((time.perf_counter() - s) * 1000)
        except Exception as e:
            print(f"Error at batch_size {batch_size}: {e}")
            break
        
        total_time = time.perf_counter() - start_total
        latencies = np.array(latencies)
        p99 = float(np.percentile(latencies, 99))
        meets_sla = p99 <= target_p99_ms
        
        if meets_sla:
            max_batch_meeting_sla = batch_size
        
        events_per_s = batch_size / (np.mean(latencies) / 1000) if latencies.size > 0 else 0
        
        results.append({
            "batch_size": batch_size,
            "p50_ms": float(np.percentile(latencies, 50)) if latencies.size > 0 else 0,
            "p99_ms": p99,
            "p999_ms": float(np.percentile(latencies, 99.9)) if latencies.size > 0 else 0,
            "events_per_s": events_per_s,
            "meets_sla": meets_sla,
        })
        
        print(f"  Batch {batch_size:3d}: p99={p99:6.1f}ms, SLA={meets_sla}, ev/s={events_per_s:.0f}")
    
    # Compute throughput at max batch meeting SLA
    max_result = next((r for r in results if r["batch_size"] == max_batch_meeting_sla), results[0])
    
    return {
        "max_batch_at_sla": max_batch_meeting_sla,
        "events_per_s": max_result["events_per_s"],
        "tokens_per_s": max_result["events_per_s"] * 128,  # assuming 128 tokens/event
        "sweep": results,
    }

def k_sweep(model: Any, prompt: str, k_values: List[int] = None) -> Dict[str, Any]:
    """Sweep K denoising steps to find knee"""
    if k_values is None:
        k_values = [2, 4, 6, 8, 10, 12, 16, 24]
    
    results = []
    print(f"Running K-sweep: {k_values}")
    
    for k in k_values:
        # Set K steps
        if hasattr(model, 'k_steps'):
            model.k_steps = k
        
        latencies = []
        for _ in range(10):  # 10 samples per K
            start = time.perf_counter()
            try:
                _ = model.extract(prompt, k_steps=k)
                latencies.append((time.perf_counter() - start) * 1000)
            except Exception as e:
                print(f"Error at K={k}: {e}")
                break
        
        if len(latencies) < 5:
            continue
            
        p50 = float(np.percentile(latencies, 50))
        results.append({
            "k": k,
            "p50_ms": p50,
            "latency_per_step": p50 / k if k > 0 else 0,
        })
        print(f"  K={k:2d}: p50={p50:7.1f}ms, per_step={p50/k:.1f}ms")
    
    # Fit linear regression for latency vs K
    if len(results) >= 3:
        from numpy import polyfit
        k_vals = [r["k"] for r in results]
        lat_vals = [r["p50_ms"] for r in results]
        slope, intercept = polyfit(k_vals, lat_vals, 1)
        r_squared = 1 - sum((l - (slope*k + intercept))**2 for k,l in zip(k_vals, lat_vals)) / \
                          sum((l - np.mean(lat_vals))**2 for l in lat_vals)
        
        return {
            "k_sweep": results,
            "linear_fit": {
                "slope_ms_per_step": float(slope),
                "intercept_ms": float(intercept),
                "r_squared": float(r_squared),
            },
            "knee_estimate": next((r["k"] for r in results if r["p50_ms"] > 1000), 8),  # heuristic
        }
    
    return {"k_sweep": results}

def main():
    parser = argparse.ArgumentParser(description="Issue #12: MDM Performance Validation")
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                       help="Model path or name")
    parser.add_argument("--n", type=int, default=500, help="Number of test events")
    parser.add_argument("--k-steps", type=int, default=8, help="Denoising steps for MDM")
    parser.add_argument("--mock", action="store_true", help="Use mock models for testing")
    parser.add_argument("--output", type=str, default="results/diffusiongemma_perf.json",
                       help="Output JSON file")
    args = parser.parse_args()
    
    print(f"=== Issue #12: MDM Performance Validation ===")
    print(f"Model: {args.model}")
    print(f"N events: {args.n}")
    print(f"K steps: {args.k_steps}")
    print(f"Mock: {args.mock}")
    
    # Load models
    if args.mock:
        print("\nUsing MOCK models for pipeline testing...")
        model = MockMDMModel(k_steps=args.k_steps)
    else:
        print(f"\nLoading {args.model}...")
        model = MDMModel(args.model)
        model.load()
        model.k_steps = args.k_steps
    
    # Load test data
    print(f"\nLoading BGL test data (n={args.n})...")
    test_data = load_bgl_test_data(args.n)
    prompts = [item["log"] for item in test_data[:min(100, len(test_data))]]  # Use 100 for speed
    
    if not prompts:
        print("No prompts loaded, using synthetic")
        prompts = [f"ERROR: Test log {i}" for i in range(50)]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Run experiments
    results = {
        "model": args.model,
        "k_steps": args.k_steps,
        "n_events": args.n,
        "note": "DiffusionGemma-26B requires CUDA 12.3+ driver. Using LLaDA-8B as MDM substitute.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    
    # 1. Latency (Table 3)
    print("\n=== Latency (Table 3) ===")
    latency_results = measure_latency(model, prompts)
    results["latency_table3"] = latency_results
    print(f"p50: {latency_results['p50_ms']:.1f}ms, p99: {latency_results['p99_ms']:.1f}ms, p99.9: {latency_results['p999_ms']:.1f}ms")
    
    # 2. Throughput (Table 4)
    print("\n=== Throughput @ 250ms SLA (Table 4) ===")
    throughput_results = measure_throughput_at_sla(model, prompts[0])
    results["throughput_table4"] = throughput_results
    print(f"Max batch @SLA: {throughput_results['max_batch_at_sla']}, ev/s: {throughput_results['events_per_s']:.0f}")
    
    # 3. K-sweep (Ablation)
    print("\n=== K-sweep (Ablation) ===")
    k_results = k_sweep(model, prompts[0])
    results["k_sweep_ablation"] = k_results
    if "linear_fit" in k_results:
        fit = k_results["linear_fit"]
        print(f"Linear fit: slope={fit['slope_ms_per_step']:.1f}ms/step, R²={fit['r_squared']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Generate markdown report
    md_report = f"""# Issue #12: MDM Performance Validation

**Model:** {args.model} (LLaDA-8B as DiffusionGemma substitute)  
**K steps:** {args.k_steps}  
**N events:** {args.n}  
**Date:** {results['timestamp']}

**Note:** DiffusionGemma-26B-A4B-it requires CUDA 12.3+ driver (we have 12.2). Using LLaDA-8B as MDM substitute, consistent with issue #3 methodology.

## Table 3: Latency (Streaming Mode)

| Metric | Value (ms) |
|--------|-----------|
| p50 | {latency_results['p50_ms']:.1f} |
| p99 | {latency_results['p99_ms']:.1f} |
| p99.9 | {latency_results['p999_ms']:.1f} |
| Mean | {latency_results['mean_ms']:.1f} |
| Std | {latency_results['std_ms']:.1f} |

## Table 4: Throughput @ 250ms p99 SLA

| Metric | Value |
|--------|-------|
| Max batch @SLA | {throughput_results['max_batch_at_sla']} |
| Events/sec | {throughput_results['events_per_s']:.0f} |
| Tokens/sec | {throughput_results['tokens_per_s']:.0f} |

## K-sweep Ablation

| K | p50 (ms) | Latency/step (ms) |
|---|----------|-------------------|
"""
    
    for r in k_results.get("k_sweep", []):
        md_report += f"| {r['k']} | {r['p50_ms']:.1f} | {r['latency_per_step']:.1f} |\n"
    
    if "linear_fit" in k_results:
        fit = k_results["linear_fit"]
        md_report += f"\n**Linear fit:** slope={fit['slope_ms_per_step']:.1f}ms/step, intercept={fit['intercept_ms']:.0f}ms, R²={fit['r_squared']:.4f}\n"
    
    md_report += f"\n---\n*Generated by `run_diffusiongemma_perf.py`*\n"
    
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    print(f"✓ Markdown report saved to {md_path}")

if __name__ == "__main__":
    main()
