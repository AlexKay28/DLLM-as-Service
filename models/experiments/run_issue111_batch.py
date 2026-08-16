#!/usr/bin/env python3
"""Issue #111 Phase 4: Batch sweep for Qwen3-30B-A3B."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
exec(open(str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patch.py")).read())

import json, time, os
import numpy as np
REPO = Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def load_bgl_events(n=300):
    lines = []
    with open(REPO / "datasets" / "splits" / "test" / "bgl.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n: break
            lines.append(line.strip())
    return lines

def run_batch_sweep():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from models.baselines.ar_model import build_chat_messages

    model_name = "Qwen/Qwen3-30B-A3B"
    print(f"Loading {model_name}...")
    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85,
              enforce_eager=True, trust_remote_code=True, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    events = load_bgl_events(300)
    prompts = []
    for line in events:
        messages = build_chat_messages(line, source="bgl", few_shot=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
        prompts.append(text)

    sp = SamplingParams(temperature=0.0, max_tokens=256, min_tokens=0)

    # Batch 1 (serial): process one at a time
    print("\n=== Batch-1 (serial) ===")
    t0 = time.time()
    for p in prompts[:20]:  # 20 events serial
        llm.generate([p], sp, use_tqdm=False)
    serial_time = time.time() - t0
    serial_per_event = serial_time * 1000 / 20
    print(f"  Serial: {serial_per_event:.1f}ms/event ({20}/{serial_time:.1f}s)")

    # Batch all: process all 300 at once
    print("\n=== Batch-all (300 concurrent) ===")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    batch_time = time.time() - t0
    batch_per_event = batch_time * 1000 / len(prompts)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_s = total_tokens / batch_time
    events_per_s = len(prompts) / batch_time
    print(f"  Batch: {batch_per_event:.1f}ms/event ({len(prompts)}/{batch_time:.1f}s)")
    print(f"  Throughput: {events_per_s:.1f} events/s, {tokens_per_s:.0f} tokens/s")

    multiplier = serial_per_event / batch_per_event
    print(f"\n=== Multiplier: {multiplier:.1f}x ===")
    print(f"  Compare: AR-3B dense = 42.8x, DG = <=1.7x")

    result = {
        "model": model_name,
        "serial_per_event_ms": round(serial_per_event, 1),
        "batch_per_event_ms": round(batch_per_event, 1),
        "batch_size": len(prompts),
        "wall_time_s": round(batch_time, 2),
        "events_per_s": round(events_per_s, 1),
        "tokens_per_s": round(tokens_per_s, 0),
        "multiplier": round(multiplier, 1),
        "comparison": {
            "AR-3B_dense": "42.8x",
            "DG-26B": "<=1.7x",
        }
    }

    out_path = REPO / "results" / "issue111_qwen3_batch_sweep.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")

if __name__ == "__main__":
    run_batch_sweep()
