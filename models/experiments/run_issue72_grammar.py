#!/usr/bin/env python3
"""Issue #72: Grammar-constrained AR-1.5B baseline."""

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

JSON_SCHEMA = {"type":"object","properties":{f:{"type":"string"} for f in FIELD_NAMES},"required":FIELD_NAMES,"additionalProperties":False}

def load_bgl_events(n=300):
    lines = []
    with open(REPO / "datasets" / "splits" / "test" / "bgl.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n: break
            lines.append(line.strip())
    return lines

def run_experiment(n_events=300):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name}...")
    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85,
              enforce_eager=True, trust_remote_code=True, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    events = load_bgl_events(n_events)
    print(f"Loaded {len(events)} BGL events")

    prompts = []
    for line in events:
        messages = build_chat_messages(line, source="bgl", few_shot=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    guided = GuidedDecodingParams(json=json.dumps(JSON_SCHEMA))
    sp = SamplingParams(temperature=0.0, max_tokens=256, min_tokens=0, guided_decoding=guided)

    # Batch mode (all at once) — measure total wall time
    print("Running grammar-constrained (batch, max=256)...")
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
        "experiment": "issue72_grammar_constrained_ar1b",
        "model": model_name, "vllm_version": "0.10.2", "torch_version": "2.8.0+cu128",
        "gpu_model": "NVIDIA A100 80GB PCIe", "n_events": len(events), "corpus": "BGL",
        "grammar": "xgrammar", "max_new_tokens": 256, "min_tokens": 0, "temperature": 0.0,
        "mode": "batch (per-event = wall_time / n)",
        "wall_time_s": round(wall_time, 2),
        "per_event_ms": round(per_event_ms, 1),
        "latency_ms": {"p50": round(per_event_ms, 1), "mean": round(per_event_ms, 1)},
        "quality": {"field_f1_9": round(f1_9, 4), "schema_valid_pct": round(schema_valid_pct, 1),
                     "per_field_f1": {k: round(v, 4) for k, v in f1_result.items()}},
        "mean_output_tokens": round(float(np.mean(token_counts)), 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = REPO / "results" / "issue72_ar1b_grammar.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")
    print(f"\n=== AR-1.5B + xgrammar ===")
    print(f"  per_event: {per_event_ms:.1f} ms (batch)")
    print(f"  9-field F1: {f1_9:.4f}")
    print(f"  schema_valid: {schema_valid_pct:.1f}%")
    print(f"  mean_output_tokens: {np.mean(token_counts):.1f}")
    print(f"  wall_time: {wall_time:.1f}s")
    print(f"\n=== Comparison ===")
    print(f"  AR-1.5B unforced (no grammar): 665.7 ms, F1 0.868, ~138 tokens")
    print(f"  AR-1.5B + grammar (this run):  {per_event_ms:.1f} ms, F1 {f1_9:.4f}, ~{np.mean(token_counts):.0f} tokens")
    print(f"  DG-26B (baseline):              378.3 ms, F1 0.852, ~172 tokens")

def _is_valid(text):
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and all(f in obj for f in FIELD_NAMES)
    except:
        return False

if __name__ == "__main__":
    run_experiment(n_events=300)
