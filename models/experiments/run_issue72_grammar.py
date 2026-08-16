#!/usr/bin/env python3
"""Issue #72: Grammar-constrained AR-1.5B baseline (simplified schema)."""

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

# Simplified schema: ALL strings (xgrammar struggles with mixed types)
JSON_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in FIELD_NAMES},
    "required": FIELD_NAMES,
    "additionalProperties": False,
}

def load_bgl_events(n=300):
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

    # Run WITHOUT grammar first (baseline)
    print("\n=== Run 1: No grammar (baseline) ===")
    sp_plain = SamplingParams(temperature=0.0, max_tokens=192, min_tokens=0)
    t0 = time.time()
    outputs_plain = llm.generate(prompts, sp_plain)
    wall_plain = time.time() - t0
    plain_texts = [out.outputs[0].text for out in outputs_plain]
    plain_tokens = [len(out.outputs[0].token_ids) for out in outputs_plain]
    plain_valid = sum(1 for t in plain_texts if _is_valid(t))

    # Run WITH grammar
    print("\n=== Run 2: With xgrammar ===")
    guided = GuidedDecodingParams(json=json.dumps(JSON_SCHEMA))
    sp_grammar = SamplingParams(temperature=0.0, max_tokens=192, min_tokens=0, guided_decoding=guided)
    t0 = time.time()
    outputs_grammar = llm.generate(prompts, sp_grammar)
    wall_grammar = time.time() - t0
    grammar_texts = [out.outputs[0].text for out in outputs_grammar]
    grammar_tokens = [len(out.outputs[0].token_ids) for out in outputs_grammar]
    grammar_valid = sum(1 for t in grammar_texts if _is_valid(t))

    # Score both
    gold_records = []
    for line in events:
        try:
            gold = json.loads(line)
            gold_records.append(Record(**{k: gold.get(k, "") for k in FIELD_NAMES}))
        except:
            gold_records.append(Record())

    nine_fields = FIELD_NAMES[:9]
    
    # Plain scoring
    plain_records = []
    for text in plain_texts:
        try:
            d = parse_json_output(text)
            plain_records.append(Record(**{k: d.get(k, "") for k in FIELD_NAMES}))
        except:
            plain_records.append(Record())
    plain_f1 = field_f1(plain_records, gold_records, fields_to_eval=nine_fields)
    plain_f1_9 = sum(plain_f1.values()) / len(plain_f1) if plain_f1 else 0.0

    # Grammar scoring
    grammar_records = []
    for text in grammar_texts:
        try:
            d = parse_json_output(text)
            grammar_records.append(Record(**{k: d.get(k, "") for k in FIELD_NAMES}))
        except:
            grammar_records.append(Record())
    grammar_f1 = field_f1(grammar_records, gold_records, fields_to_eval=nine_fields)
    grammar_f1_9 = sum(grammar_f1.values()) / len(grammar_f1) if grammar_f1 else 0.0

    # Sample outputs for debugging
    print("\n=== Sample outputs (first 3) ===")
    for i in range(min(3, len(grammar_texts))):
        print(f"\n--- Event {i} (grammar) ---")
        print(grammar_texts[i][:500])
        print(f"Valid: {_is_valid(grammar_texts[i])}")

    result = {
        "experiment": "issue72_grammar_vs_plain",
        "model": model_name, "vllm_version": "0.10.2", "torch_version": "2.8.0+cu128",
        "gpu_model": "NVIDIA A100 80GB PCIe", "n_events": len(events), "corpus": "BGL",
        "plain": {
            "wall_time_s": round(wall_plain, 2),
            "per_event_ms": round(wall_plain * 1000 / len(events), 1),
            "mean_tokens": round(float(np.mean(plain_tokens)), 1),
            "schema_valid_pct": round(100.0 * plain_valid / len(events), 1),
            "f1_9": round(plain_f1_9, 4),
        },
        "grammar": {
            "wall_time_s": round(wall_grammar, 2),
            "per_event_ms": round(wall_grammar * 1000 / len(events), 1),
            "mean_tokens": round(float(np.mean(grammar_tokens)), 1),
            "schema_valid_pct": round(100.0 * grammar_valid / len(events), 1),
            "f1_9": round(grammar_f1_9, 4),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = REPO / "results" / "issue72_ar1b_grammar.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")
    print(f"\n=== RESULTS ===")
    print(f"{'':30s} {'Plain':>10s} {'Grammar':>10s}")
    print(f"{'Per-event (ms)':30s} {result['plain']['per_event_ms']:>10.1f} {result['grammar']['per_event_ms']:>10.1f}")
    print(f"{'Mean tokens':30s} {result['plain']['mean_tokens']:>10.1f} {result['grammar']['mean_tokens']:>10.1f}")
    print(f"{'Schema valid %':30s} {result['plain']['schema_valid_pct']:>10.1f} {result['grammar']['schema_valid_pct']:>10.1f}")
    print(f"{'9-field F1':30s} {result['plain']['f1_9']:>10.4f} {result['grammar']['f1_9']:>10.4f}")
    print(f"\n=== Comparison with paper ===")
    print(f"  AR-1.5B serial unforced: 665.7ms, F1 0.868, ~138 tokens")
    print(f"  DG-26B serial:           378.3ms, F1 0.852, ~172 tokens")

if __name__ == "__main__":
    run_experiment(n_events=300)
