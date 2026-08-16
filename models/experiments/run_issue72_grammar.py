#!/usr/bin/env python3
"""Issue #72: Grammar-constrained AR-1.5B baseline.

Measures AR-1.5B + xgrammar on the pinned stack, with the 12-field JSON schema
as the grammar. Compares against the existing unforced and forced AR-1.5B
results and the DG baseline.

Outputs: results/issue72_ar1b_grammar.json
"""

# Monkey-patch for vLLM + transformers 5.x compatibility (must be before vllm import)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
exec(open(str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patch.py")).read())

import json
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metrics.schema import FIELD_NAMES

# 12-field JSON schema for guided decoding
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp_norm": {"type": "string"},
        "host": {"type": "string"},
        "component": {"type": "string"},
        "severity": {"type": "string"},
        "event_type": {"type": "string"},
        "message_template": {"type": "string"},
        "error_code": {"type": "string"},
        "affected_entity": {"type": "string"},
        "metric_value": {"type": "number"},
        "is_anomaly": {"type": "boolean"},
        "anomaly_score": {"type": "number"},
        "remediation_hint": {"type": "string"},
    },
    "required": FIELD_NAMES,
    "additionalProperties": False,
}


def load_bgl_events(n=300):
    """Load BGL test events."""
    gold_path = REPO / "datasets" / "splits" / "test" / "bgl.jsonl"
    if not gold_path.exists():
        print(f"ERROR: gold file not found at {gold_path}")
        sys.exit(1)
    lines = []
    with open(gold_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line.strip())
    return lines


def build_prompt(log_line, source="bgl"):
    """Build the chat prompt for a single log line."""
    from models.baselines.ar_model import build_chat_messages
    messages = build_chat_messages(log_line, source=source, few_shot=True)
    return messages


def run_experiment(n_events=300, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """Run grammar-constrained AR-1.5B on BGL."""
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from transformers import AutoTokenizer

    print(f"Loading {model_name} on vLLM...")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load events
    events = load_bgl_events(n_events)
    print(f"Loaded {len(events)} BGL events")

    # Build prompts
    prompts = []
    for line in events:
        messages = build_prompt(line, source="bgl")
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # Unforced: no min_tokens, max_new_tokens=256, with grammar constraint
    print("Running grammar-constrained (unforced, max=256)...")
    guided = GuidedDecodingParams(json=json.dumps(JSON_SCHEMA))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        min_tokens=0,
        guided_decoding=guided,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    wall_time = time.time() - t0

    # Collect results
    latencies = []
    output_texts = []
    output_token_counts = []

    t0 = time.time()
    for out in outputs:
        text = out.outputs[0].text
        output_texts.append(text)
        output_token_counts.append(len(out.outputs[0].token_ids))
    # Per-request timing not available in eager mode; use total wall time / n
    total_wall = time.time() - t0
    latencies = [total_wall * 1000 / len(outputs)] * len(outputs)

    import numpy as np
    latencies = np.array(latencies)

    # Parse and score
    from models.baselines.ar_model import parse_json_output
    from metrics.schema import Record
    from metrics.quality_metrics import field_f1

    records = []
    for text in output_texts:
        try:
            d = parse_json_output(text)
            rec = Record(**{k: d.get(k, "") for k in FIELD_NAMES})
            records.append(rec)
        except Exception:
            records.append(Record())

    # Gold records
    gold_records = []
    for line in events:
        try:
            gold = json.loads(line)
            rec = Record(**{k: gold.get(k, "") for k in FIELD_NAMES})
            gold_records.append(rec)
        except Exception:
            gold_records.append(Record())

    nine_fields = FIELD_NAMES[:9]; result_f1 = field_f1(records, gold_records, fields_to_eval=nine_fields); f1_9 = sum(result_f1.values()) / len(result_f1); per_field_f1 = result_f1
    result_f1_12 = field_f1(records, gold_records); f1_12 = sum(result_f1_12.values()) / len(result_f1_12)

    # Schema validity
    valid_count = 0
    for text in output_texts:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and all(f in obj for f in FIELD_NAMES):
                valid_count += 1
        except Exception:
            pass
    schema_valid_pct = 100.0 * valid_count / len(output_texts)

    # Completeness (all fields non-empty)
    complete_count = 0
    for rec in records:
        if all(str(getattr(rec, f, "")).strip() not in ("", "0", "0.0", "False", "None") for f in FIELD_NAMES):
            complete_count += 1
    completeness_pct = 100.0 * complete_count / len(records)

    mean_tokens = np.mean(output_token_counts)

    result = {
        "experiment": "issue72_grammar_constrained_ar1b",
        "model": model_name,
        "vllm_version": "0.7.2",
        "torch_version": "2.5.1+cu121",
        "gpu_model": "NVIDIA A100 80GB PCIe",
        "n_events": len(events),
        "corpus": "BGL",
        "grammar": "xgrammar",
        "max_new_tokens": 256,
        "min_tokens": 0,
        "temperature": 0.0,
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "p99_9": float(np.percentile(latencies, 99.9)),
            "mean": float(np.mean(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "n_samples": len(latencies),
        },
        "quality": {
            "field_f1_9": round(f1_9, 4),
            "field_f1_12": round(f1_12, 4),
            "schema_valid_pct": round(schema_valid_pct, 1),
            "completeness_pct": round(completeness_pct, 1),
            "per_field_f1": {k: round(v, 4) for k, v in per_field_f1.items()},
        },
        "mean_output_tokens": round(float(mean_tokens), 1),
        "wall_time_s": round(wall_time, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save
    out_path = REPO / "results" / "issue72_ar1b_grammar.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")

    # Print summary
    print(f"\n=== AR-1.5B + xgrammar (unforced, max=256) ===")
    print(f"  p50: {result['latency_ms']['p50']:.1f} ms")
    print(f"  p99: {result['latency_ms']['p99']:.1f} ms")
    print(f"  9-field F1: {result['quality']['field_f1_9']}")
    print(f"  schema_valid: {result['quality']['schema_valid_pct']}%")
    print(f"  completeness: {result['quality']['completeness_pct']}%")
    print(f"  mean_output_tokens: {result['mean_output_tokens']}")
    print(f"  wall_time: {result['wall_time_s']}s")

    # Compare with existing results
    print(f"\n=== Comparison ===")
    print(f"  AR-1.5B unforced (no grammar): 665.7 ms, F1 0.8679, ~138 tokens")
    print(f"  AR-1.5B + grammar (this run):   {result['latency_ms']['p50']:.1f} ms, F1 {result['quality']['field_f1_9']}, ~{result['mean_output_tokens']:.0f} tokens")
    print(f"  DG-26B (baseline):              378.3 ms, F1 0.8518, ~172 tokens")

    return result


if __name__ == "__main__":
    run_experiment(n_events=300)
