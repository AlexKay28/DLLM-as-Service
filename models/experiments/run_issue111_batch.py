#!/usr/bin/env python3
"""Issue #111 / #112: batch sweep for Qwen3-30B-A3B (AR-30B).

Produces ``results/issue111_qwen3_batch_sweep.json`` -- the AR-30B
batch-1 -> peak multiplier cited by ``paper/figures/make_figures.py`` and the
macros comment block. Fixed for issue #112:

  - model input = ``bgl.raw`` (was ``bgl.jsonl`` -- the gold file; the
    build_chat_messages guard now raises on this class of leak)
  - no ``enforce_eager``; ``enable_prefix_caching=True``
  - serial arm is true serial (one prompt per ``generate``), like the anchors
  - provenance block via ``capture_provenance``

Pinned stack: vLLM 0.25.1 / torch 2.11.0 / CUDA 13.0 (must match anchors).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
exec(open(str(REPO / "scripts" / "vllm_patch.py")).read())

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from models.baselines.ar_model import build_chat_messages
from models.baselines.provenance import capture_provenance

SPLIT_DIR = REPO / "datasets" / "splits" / "test"


def load_log_lines(path, n=None):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.rstrip("\n")
            if line:
                lines.append(line)
    return lines


def run_batch_sweep(n_events=300, serial_n=20, max_tokens=256):
    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen3-30B-A3B"
    print(f"Loading {model_name}...", flush=True)
    llm = LLM(
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=4096,
        enable_prefix_caching=True, trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    log_lines = load_log_lines(str(SPLIT_DIR / "bgl.raw"), n=n_events)
    print(f"Loaded {len(log_lines)} BGL events", flush=True)

    prompts = []
    for line in log_lines:
        messages = build_chat_messages(line, source="bgl", few_shot=True)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=0)

    # Batch-1 (serial): one prompt per generate call.
    print(f"\n=== Batch-1 (serial, n={serial_n}) ===", flush=True)
    serial_latencies = []
    t0 = time.time()
    for p in prompts[:serial_n]:
        gt0 = time.perf_counter()
        llm.generate([p], sp, use_tqdm=False)
        serial_latencies.append((time.perf_counter() - gt0) * 1000.0)
    serial_time = time.time() - t0
    serial_per_event = statistics.fmean(serial_latencies)
    print(f"  Serial: {serial_per_event:.1f}ms/event "
          f"({serial_n}/{serial_time:.1f}s)")

    # Batch-all: all events concurrently.
    print(f"\n=== Batch-all ({len(prompts)} concurrent) ===", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    batch_time = time.time() - t0
    batch_per_event = batch_time * 1000 / len(outputs)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_s = total_tokens / batch_time
    events_per_s = len(prompts) / batch_time
    print(f"  Batch: {batch_per_event:.1f}ms/event "
          f"({len(prompts)}/{batch_time:.1f}s)")
    print(f"  Throughput: {events_per_s:.1f} events/s, "
          f"{tokens_per_s:.0f} tokens/s")

    multiplier = serial_per_event / batch_per_event
    print(f"\n=== Multiplier: {multiplier:.1f}x ===")
    print(f"  Compare: AR-3B dense = 42.8x, DG = <=1.7x")

    prov = capture_provenance(
        model_name=model_name, model_type="ar30b", backend="vllm",
        max_new_tokens=max_tokens, parser_policy="first_object_wins",
        scoring_fields="n/a (throughput)",
    )

    result = {
        "experiment": "issue111_ar30b_batch_sweep",
        "model": model_name,
        "model_type": "ar30b",
        "corpus": "BGL",
        "n_events": len(prompts),
        "serial_n": serial_n,
        "max_new_tokens": max_tokens,
        "temperature": 0.0,
        "serial_per_event_ms": round(serial_per_event, 1),
        "serial_p50_ms": round(statistics.median(serial_latencies), 1),
        "batch_per_event_ms": round(batch_per_event, 1),
        "batch_size": len(prompts),
        "wall_time_s": round(batch_time, 2),
        "events_per_s": round(events_per_s, 1),
        "tokens_per_s": round(tokens_per_s, 0),
        "multiplier": round(multiplier, 1),
        "comparison": {
            "AR-3B_dense": "42.8x",
            "DG-26B": "<=1.7x",
        },
        "provenance": prov.to_dict(),
    }

    out_path = REPO / "results" / "issue111_qwen3_batch_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--serial-n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    run_batch_sweep(n_events=args.n, serial_n=args.serial_n,
                    max_tokens=args.max_tokens)
