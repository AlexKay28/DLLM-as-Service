#!/usr/bin/env python3
"""Issue #111: Qwen3-30B-A3B MoE-matched AR baseline."""

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

def load_bgl_events(n=300):
    lines = []
    with open(REPO / "datasets" / "splits" / "test" / "bgl.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n: break
            lines.append(line.strip())
    return lines

def run_experiment(n_events=300):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen3-30B-A3B"
    print(f"Loading {model_name}...")
    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85,
              enforce_eager=True, trust_remote_code=True, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    events = load_bgl_events(n_events)
    print(f"Loaded {len(events)} BGL events")

    prompts = []
    for line in events:
        messages = build_chat_messages(line, source="bgl", few_shot=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(text)

    # Run at multiple output lengths
    results = {}
    for max_tokens in [256]:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=0)
        print(f"\n=== Running max_tokens={max_tokens} ===")
        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        wall_time = time.time() - t0

        output_texts = [out.outputs[0].text for out in outputs]
        token_counts = [len(out.outputs[0].token_ids) for out in outputs]
        per_event_ms = wall_time * 1000 / len(outputs)

        # Parse and score
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

        valid_count = sum(1 for t in output_texts if _is_valid(t))
        schema_valid_pct = 100.0 * valid_count / len(output_texts)

        result = {
            "model": model_name, "vllm_version": "0.11.0", "torch_version": "2.8.0+cu128",
            "gpu_model": "NVIDIA A100 80GB PCIe", "n_events": len(events), "corpus": "BGL",
            "max_new_tokens": max_tokens, "min_tokens": 0, "temperature": 0.0,
            "wall_time_s": round(wall_time, 2),
            "per_event_ms": round(per_event_ms, 1),
            "mean_output_tokens": round(float(np.mean(token_counts)), 1),
            "field_f1_9": round(f1_9, 4),
            "schema_valid_pct": round(schema_valid_pct, 1),
            "per_field_f1": {k: round(v, 4) for k, v in f1_result.items()},
        }
        results[f"L{max_tokens}"] = result

        print(f"  per_event: {per_event_ms:.1f}ms")
        print(f"  F1_9: {f1_9:.4f}")
        print(f"  schema_valid: {schema_valid_pct:.1f}%")
        print(f"  mean_tokens: {np.mean(token_counts):.1f}")
        print(f"  wall_time: {wall_time:.1f}s")

    # Save
    out_path = REPO / "results" / "issue111_qwen3_30b_a3b.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {out_path}")

    print(f"\n=== COMPARISON ===")
    print(f"  AR-30B (Qwen3 MoE):   {results['L256']['per_event_ms']:.1f}ms, F1 {results['L256']['field_f1_9']:.4f}")
    print(f"  AR-3B (Qwen2.5 dense): 665.7ms, F1 0.868 (serial)")
    print(f"  DG-26B-A4B:             378.3ms, F1 0.852 (serial)")

def _is_valid(text):
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and all(f in obj for f in FIELD_NAMES)
    except:
        return False

if __name__ == "__main__":
    run_experiment(n_events=300)
