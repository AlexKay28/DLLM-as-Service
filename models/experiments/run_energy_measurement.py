#!/usr/bin/env python3
"""#5: Energy measurement at the compute-matched pair.

Measures board-level GPU power during AR-3B and DG inference using
nvidia-smi dmon, subtracts idle baseline, computes J/event.

Usage:
    .venv-dg/bin/python models/experiments/run_energy_measurement.py
"""

import json
import subprocess
import sys
import time
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record
from metrics.quality_metrics import field_f1
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.provenance import capture_provenance

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
OUT_DIR = _REPO_ROOT / "results"


def measure_idle_power(duration_s=10):
    """Measure idle GPU power for baseline."""
    print(f"Measuring idle power ({duration_s}s)...", flush=True)
    proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "p", "-d", "1", "-c", str(duration_s)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    stdout, _ = proc.communicate()
    powers = []
    for line in stdout.decode().strip().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                power = float(parts[3])  # power in W
                powers.append(power)
            except ValueError:
                continue
    idle = statistics.fmean(powers) if powers else 0
    print(f"  Idle power: {idle:.1f} W (n={len(powers)})", flush=True)
    return idle


def measure_power_during_inference(label, model_name, log_lines, gold, L=256,
                                     is_dg=False, k_steps=8, n=100):
    """Run inference while measuring GPU power."""
    from vllm import LLM, SamplingParams

    print(f"\n=== {label} ===", flush=True)

    llm_kwargs = dict(
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=4096,
        enable_prefix_caching=True,
    )
    if is_dg:
        llm_kwargs["diffusion_config"] = {
            "canvas_length": 256, "max_denoising_steps": k_steps}

    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=L, min_tokens=L, temperature=0.0,
                        repetition_penalty=1.0)

    prompts = []
    for line in log_lines[:n]:
        msgs = build_chat_messages(line, source="bgl", few_shot=True)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True))

    # Start power monitoring
    power_proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "p", "-d", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    # Run inference
    latencies = []
    records = []
    t_start = time.perf_counter()
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - t0) * 1000)
        records.append(_dict_to_record(parse_json_output(outs[0].outputs[0].text)))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(prompts)} p50={statistics.median(latencies):.0f}ms",
                  flush=True)

    inference_duration = time.perf_counter() - t_start

    # Stop power monitoring
    power_proc.terminate()
    stdout, _ = power_proc.communicate(timeout=5)

    # Parse power readings
    powers = []
    for line in stdout.decode().strip().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                powers.append(float(parts[3]))
            except ValueError:
                continue

    avg_power = statistics.fmean(powers) if powers else 0
    total_energy_j = avg_power * inference_duration
    energy_per_event = total_energy_j / len(records) if records else 0

    lat = latency_percentiles(latencies)
    f1_res = field_f1(records, gold[:len(records)])

    prov = capture_provenance(
        model_name=model_name,
        model_type="dg" if is_dg else "ar3b",
        backend="vllm",
        max_new_tokens=L,
        parser_policy="first_object_wins",
        scoring_fields="9",
    )

    result = {
        "label": label,
        "model": model_name,
        "n_events": len(records),
        "L": L,
        "inference_duration_s": round(inference_duration, 2),
        "avg_power_w": round(avg_power, 1),
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_event_j": round(energy_per_event, 1),
        "p50_ms": round(lat["p50_ms"], 1),
        "p99_ms": round(lat["p99_ms"], 1),
        "field_f1_9": round(f1_res["macro_field_f1_9"], 4),
        "n_power_samples": len(powers),
        "provenance": prov.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"  p50={lat['p50_ms']:.0f}ms  avg_power={avg_power:.1f}W  "
          f"energy/event={energy_per_event:.1f}J  F1={f1_res['macro_field_f1_9']:.4f}",
          flush=True)

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--L", type=int, default=256)
    args = ap.parse_args()

    log_lines = []
    with open(SPLIT_DIR / "bgl.raw") as f:
        for i, line in enumerate(f):
            if i >= args.n: break
            line = line.strip()
            if line: log_lines.append(line)
    gold = []
    with open(SPLIT_DIR / "bgl.jsonl") as f:
        for i, line in enumerate(f):
            if i >= args.n: break
            line = line.strip()
            if line: gold.append(Record(fields=json.loads(line)))
    print(f"Loaded {len(log_lines)} events", flush=True)

    # Idle baseline
    idle_power = measure_idle_power(10)

    results = {"idle_power_w": round(idle_power, 1), "measurements": []}

    # AR-3B
    ar_result = measure_power_during_inference(
        "AR-3B L=256", "Qwen/Qwen2.5-3B-Instruct",
        log_lines, gold, L=args.L, n=args.n)
    ar_result["idle_subtracted_power_w"] = round(ar_result["avg_power_w"] - idle_power, 1)
    ar_result["idle_subtracted_energy_per_event_j"] = round(
        (ar_result["avg_power_w"] - idle_power) * ar_result["inference_duration_s"] / ar_result["n_events"], 1)
    results["measurements"].append(ar_result)

    # DG (K=8)
    dg_result = measure_power_during_inference(
        "DG-26B K=8 L=256", "google/diffusiongemma-26B-A4B-it",
        log_lines, gold, L=args.L, is_dg=True, k_steps=8, n=args.n)
    dg_result["idle_subtracted_power_w"] = round(dg_result["avg_power_w"] - idle_power, 1)
    dg_result["idle_subtracted_energy_per_event_j"] = round(
        (dg_result["avg_power_w"] - idle_power) * dg_result["inference_duration_s"] / dg_result["n_events"], 1)
    results["measurements"].append(dg_result)

    # Summary
    print("\n=== Energy Summary ===")
    print(f"  Idle power: {idle_power:.1f} W")
    for m in results["measurements"]:
        print(f"  {m['label']}:")
        print(f"    avg_power={m['avg_power_w']:.1f}W  net_power={m['idle_subtracted_power_w']:.1f}W")
        print(f"    energy/event={m['energy_per_event_j']:.1f}J  net={m['idle_subtracted_energy_per_event_j']:.1f}J")
        print(f"    p50={m['p50_ms']:.0f}ms  F1={m['field_f1_9']:.4f}")

    out_path = OUT_DIR / "issue_energy_measurement.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
