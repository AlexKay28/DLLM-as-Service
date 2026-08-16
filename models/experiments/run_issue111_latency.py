#!/usr/bin/env python3
"""Issue #111 Phase 3: Latency sweep for Qwen3-30B-A3B."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
exec(open(str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patch.py")).read())

import json, time, os
import numpy as np
REPO = Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from metrics.schema import FIELD_NAMES
from models.baselines.ar_model import build_chat_messages, parse_json_output
from metrics.schema import Record
from metrics.quality_metrics import field_f1

def load_bgl_events(n=100):
    lines = []
    with open(REPO / "datasets" / "splits" / "test" / "bgl.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n: break
            lines.append(line.strip())
    return lines

def _is_valid(text):
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and all(f in obj for f in FIELD_NAMES)
    except:
        return False

def run_experiment():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen3-30B-A3B"
    print(f"Loading {model_name}...")
    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85,
              enforce_eager=True, trust_remote_code=True, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    events = load_bgl_events(100)
    print(f"Loaded {len(events)} BGL events")

    prompts = []
    for line in events:
        messages = build_chat_messages(line, source="bgl", few_shot=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
        prompts.append(text)

    # Sweep at multiple output lengths
    results = {}
    for max_tokens in [32, 128, 256, 512, 1024]:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=0)
        print(f"\n=== max_tokens={max_tokens} ===")
        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        wall_time = time.time() - t0
        per_event_ms = wall_time * 1000 / len(outputs)

        token_counts = [len(out.outputs[0].token_ids) for out in outputs]
        output_texts = [out.outputs[0].text for out in outputs]
        valid = sum(1 for t in output_texts if _is_valid(t))

        # Score F1
        records = []
        for text in output_texts:
            try:
                d = parse_json_output(text)
                records.append(Record(**{k: d.get(k, "") for k in FIELD_NAMES}))
            except:
                records.append(Record())

        gold_records = []
        for line in events:
            try:
                gold = json.loads(line)
                gold_records.append(Record(**{k: gold.get(k, "") for k in FIELD_NAMES}))
            except:
                gold_records.append(Record())

        nine_fields = FIELD_NAMES[:9]
        f1_result = field_f1(records, gold_records, fields_to_eval=nine_fields)
        f1_9 = sum(f1_result.values()) / len(f1_result) if f1_result else 0.0

        result = {
            "max_tokens": max_tokens,
            "wall_time_s": round(wall_time, 2),
            "per_event_ms": round(per_event_ms, 1),
            "mean_output_tokens": round(float(np.mean(token_counts)), 1),
            "field_f1_9": round(f1_9, 4),
            "schema_valid_pct": round(100.0 * valid / len(outputs), 1),
        }
        results[f"L{max_tokens}"] = result
        print(f"  per_event: {per_event_ms:.1f}ms, tokens: {np.mean(token_counts):.1f}, F1: {f1_9:.4f}, valid: {valid}%")

    # Fit cost law: T = a + b * L
    Ls = [results[f"L{mt}"]["mean_output_tokens"] for mt in [32,128,256,512,1024]]
    Ts = [results[f"L{mt}"]["per_event_ms"] for mt in [32,128,256,512,1024]]
    # Linear fit
    coeffs = np.polyfit(Ls, Ts, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    print(f"\n=== Cost law fit: T = {intercept:.1f} + {slope:.2f} * L ===")
    print(f"  intercept (a): {intercept:.1f} ms")
    print(f"  slope (b): {slope:.2f} ms/token")
    print(f"  Compare: AR-3B dense slope = 6.3 ms/token")
    print(f"  Compare: AR-1.5B dense slope = 3.79 ms/token")

    # Crossover with DG
    dg_base, dg_step = 378.8, 119.5
    if slope > 0:
        L_star = (dg_base - intercept) / slope
        print(f"  Crossover L* = ({dg_base} - {intercept:.1f}) / {slope:.2f} = {L_star:.0f} tokens")
        print(f"  Compare: old L* = 59 tokens (with AR-3B dense)")

    results["cost_law"] = {
        "intercept_ms": round(intercept, 1),
        "slope_ms_per_token": round(float(slope), 2),
        "crossover_L_star": round(float(L_star), 0) if slope > 0 else None,
    }

    out_path = REPO / "results" / "issue111_qwen3_latency_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {out_path}")

if __name__ == "__main__":
    run_experiment()
