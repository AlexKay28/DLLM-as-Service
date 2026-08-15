#!/usr/bin/env python3
"""#7 / #59 Stage 1: Infilling experiment — suffix byte-identity.

Tests whether bidirectional conditioning gives DG an advantage AR cannot
match: when filling a gap with fixed right-context (suffix), DG preserves
the suffix by construction. AR must regenerate it from the prompt.

Stage 1 only: suffix byte-identity rate, infilled-span accuracy, latency.
S ∈ {0, 32, 128, 512} suffix tokens.

Usage:
    .venv-dg/bin/python models/experiments/run_infilling.py
"""

import json
import sys
import time
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES
from metrics.quality_metrics import field_f1
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output, SYSTEM_PROMPT
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.provenance import capture_provenance

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
OUT_DIR = _REPO_ROOT / "results"


def load_events(n=50):
    lines = []
    gold = []
    with open(SPLIT_DIR / "bgl.raw") as f:
        for i, line in enumerate(f):
            if i >= n: break
            line = line.strip()
            if line: lines.append(line)
    with open(SPLIT_DIR / "bgl.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n: break
            line = line.strip()
            if line: gold.append(Record(fields=json.loads(line)))
    return lines, gold


def build_infilling_prompt(gold_record, suffix_tokens, tokenizer, model_type):
    """Build an infilling prompt: given a JSON record, mask the middle,
    keep the suffix fixed, ask the model to fill the masked span.

    For AR: give prefix + suffix in prompt, ask to regenerate the full record.
    For DG: mask the middle tokens in the canvas, keep suffix clamped.
    """
    import json as _json
    record_json = _json.dumps(gold_record.fields, indent=2)

    # Split the record into prefix (first 3 fields) and suffix (last fields)
    fields = list(gold_record.fields.items())
    n_suffix = min(suffix_tokens, len(fields))
    prefix_fields = fields[:len(fields) - n_suffix]
    suffix_fields = fields[len(fields) - n_suffix:]

    prefix_json = _json.dumps(dict(prefix_fields), indent=2)
    suffix_json = _json.dumps(dict(suffix_fields), indent=2)

    if model_type == "ar":
        # AR: give prefix + suffix, ask to produce the full record
        prompt = f"""{SYSTEM_PROMPT}

Given the prefix and suffix of a JSON record, produce the complete record.

Prefix:
{prefix_json}

Suffix (must appear byte-identical in your output):
{suffix_json}

Produce the complete JSON record:"""
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": "Complete the record."}]
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
    else:
        # DG: build the record with a mask in the middle
        masked_record = dict(prefix_fields)
        for k, v in suffix_fields:
            masked_record[k] = "<MASK>"
        prompt = f"""{SYSTEM_PROMPT}

Complete the masked fields in this JSON record. The suffix fields are given
as hints — they must appear in your output.

{json.dumps(masked_record, indent=2)}

Output the complete JSON record:"""
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": "Complete the record."}]
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)


def measure_suffix_identity(pred_text, suffix_json):
    """Check if the suffix appears byte-identical in the prediction."""
    try:
        pred = json.loads(parse_json_output(pred_text))
        suffix = json.loads(suffix_json)
        if not isinstance(pred, dict) or not isinstance(suffix, dict):
            return 0.0
        matches = 0
        total = 0
        for k, v in suffix.items():
            total += 1
            if k in pred and str(pred[k]) == str(v):
                matches += 1
        return matches / total if total > 0 else 1.0
    except:
        return 0.0


def run_ar_infilling(llm, tokenizer, gold_records, suffix_sizes):
    """Run AR infilling: prefix + suffix in prompt, regenerate full record."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=256, min_tokens=128, temperature=0.0,
                        repetition_penalty=1.0)

    results = {}
    for S in suffix_sizes:
        print(f"  AR suffix={S}...", flush=True)
        latencies = []
        records = []
        identity_scores = []

        for i, gold_rec in enumerate(gold_records):
            fields = list(gold_rec.fields.items())
            n_suffix = min(S, len(fields))
            suffix_fields = fields[len(fields) - n_suffix:]
            suffix_json = json.dumps(dict(suffix_fields), indent=2)

            prompt = build_infilling_prompt(gold_rec, S, tokenizer, "ar")

            t0 = time.perf_counter()
            outs = llm.generate([prompt], sp, use_tqdm=False)
            latencies.append((time.perf_counter() - t0) * 1000)
            text = outs[0].outputs[0].text
            rec = _dict_to_record(parse_json_output(text))
            records.append(rec)

            identity = measure_suffix_identity(text, suffix_json)
            identity_scores.append(identity)

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(gold_records)} avg_id={statistics.mean(identity_scores):.3f}",
                      flush=True)

        lat = latency_percentiles(latencies)
        f1_res = field_f1(records, gold_records[:len(records)])
        results[f"S={S}"] = {
            "n": len(records),
            "p50_ms": round(lat["p50_ms"], 1),
            "p99_ms": round(lat["p99_ms"], 1),
            "field_f1_9": round(f1_res["macro_field_f1_9"], 4),
            "suffix_identity_rate": round(statistics.mean(identity_scores), 4),
            "identity_min": round(min(identity_scores), 4),
            "identity_max": round(max(identity_scores), 4),
        }
        print(f"    p50={lat['p50_ms']:.0f}ms  F1={f1_res['macro_field_f1_9']:.4f}  "
              f"identity={statistics.mean(identity_scores):.4f}", flush=True)

    return results


def run_dg_infilling(llm, tokenizer, gold_records, suffix_sizes):
    """Run DG infilling: mask middle fields, keep suffix as hint."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=256, min_tokens=256, temperature=0.0)

    results = {}
    for S in suffix_sizes:
        print(f"  DG suffix={S}...", flush=True)
        latencies = []
        records = []
        identity_scores = []

        for i, gold_rec in enumerate(gold_records):
            fields = list(gold_rec.fields.items())
            n_suffix = min(S, len(fields))
            suffix_fields = fields[len(fields) - n_suffix:]
            suffix_json = json.dumps(dict(suffix_fields), indent=2)

            prompt = build_infilling_prompt(gold_rec, S, tokenizer, "dg")

            t0 = time.perf_counter()
            outs = llm.generate([prompt], sp, use_tqdm=False)
            latencies.append((time.perf_counter() - t0) * 1000)
            text = outs[0].outputs[0].text
            rec = _dict_to_record(parse_json_output(text))
            records.append(rec)

            identity = measure_suffix_identity(text, suffix_json)
            identity_scores.append(identity)

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(gold_records)} avg_id={statistics.mean(identity_scores):.3f}",
                      flush=True)

        lat = latency_percentiles(latencies)
        f1_res = field_f1(records, gold_records[:len(records)])
        results[f"S={S}"] = {
            "n": len(records),
            "p50_ms": round(lat["p50_ms"], 1),
            "p99_ms": round(lat["p99_ms"], 1),
            "field_f1_9": round(f1_res["macro_field_f1_9"], 4),
            "suffix_identity_rate": round(statistics.mean(identity_scores), 4),
            "identity_min": round(min(identity_scores), 4),
            "identity_max": round(max(identity_scores), 4),
        }
        print(f"    p50={lat['p50_ms']:.0f}ms  F1={f1_res['macro_field_f1_9']:.4f}  "
              f"identity={statistics.mean(identity_scores):.4f}", flush=True)

    return results


def main():
    suffix_sizes = [0, 2, 4, 8]
    n_events = 50

    log_lines, gold_records = load_events(n_events)
    print(f"Loaded {len(gold_records)} events", flush=True)

    from vllm import LLM

    all_results = {"experiment": "issue59_infilling_stage1",
                   "n_events": n_events, "suffix_sizes": suffix_sizes}

    # AR-3B
    print("\n=== AR-3B ===", flush=True)
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    ar_results = run_ar_infilling(llm, tok, gold_records, suffix_sizes)
    all_results["ar3b"] = ar_results
    prov_ar = capture_provenance(model_name="Qwen/Qwen2.5-3B-Instruct",
                                  model_type="ar3b", backend="vllm",
                                  max_new_tokens=256,
                                  parser_policy="first_object_wins",
                                  scoring_fields="9")
    all_results["ar3b_provenance"] = prov_ar.to_dict()
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # DG
    print("\n=== DG-26B (K=8) ===", flush=True)
    llm = LLM(model="google/diffusiongemma-26B-A4B-it", dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096,
              enable_prefix_caching=True,
              diffusion_config={"canvas_length": 256, "max_denoising_steps": 8})
    tok = llm.get_tokenizer()
    dg_results = run_dg_infilling(llm, tok, gold_records, suffix_sizes)
    all_results["dg"] = dg_results
    prov_dg = capture_provenance(model_name="google/diffusiongemma-26B-A4B-it",
                                  model_type="dg", backend="vllm",
                                  max_new_tokens=256,
                                  parser_policy="first_object_wins",
                                  scoring_fields="9")
    all_results["dg_provenance"] = prov_dg.to_dict()
    all_results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    out_path = OUT_DIR / "issue59_infilling.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")

    # Summary
    print("\n=== Infilling Summary ===")
    print(f"{'Model':>6} {'S':>4} {'p50':>7} {'F1_9':>7} {'identity':>9}")
    for model_label, results_key in [("AR-3B", "ar3b"), ("DG", "dg")]:
        for s_key, r in all_results[results_key].items():
            print(f"{model_label:>6} {s_key:>4} {r['p50_ms']:7.0f} "
                  f"{r['field_f1_9']:7.4f} {r['suffix_identity_rate']:9.4f}")


if __name__ == "__main__":
    main()
