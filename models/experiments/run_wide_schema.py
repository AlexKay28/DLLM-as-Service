#!/usr/bin/env python3
"""
Issue #19 (App C): wide-schema consistency vs latency experiment.

Runs both DG and AR-3B at schema levels L12/L25/L50/L100, measures
Field-F1, schema validity, consistency, and p50 latency at each level.

The regression anchor: L12 must reproduce #13's numbers (F1≈0.79 DG,
0.82 AR).  The key test (P-C2): does DG consistency degrade monotonically
with F, faster than AR's?

Usage:
    python models/experiments/run_wide_schema.py \
        --model google/diffusiongemma-26B-A4B-it \
        --levels 12 25 50 100 --n 200
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES
from metrics.quality_metrics import field_f1, schema_validity, consistency
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.wide_schema import (
    derive_fields, field_names_for_level, build_wide_chat_messages)

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"


def load_lines(path, n):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_gold(path, n):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                records.append(Record(fields=json.loads(line)))
    return records


def summarize(values):
    out = dict(latency_percentiles(values))
    if values:
        out["mean_ms"] = float(statistics.fmean(values))
    out["n_samples"] = len(values)
    return out


def run_level(llm, sampling_params, prompts, gold_12, raw_lines, level,
              save_predictions=False):
    """Run extraction at a given schema level; measure quality + latency."""
    fields = field_names_for_level(level)
    latencies = []
    records = []
    out_tokens = []
    raw_texts = []

    print(f"  streaming {len(prompts)} events at L{level} "
          f"({len(fields)} fields)...", flush=True)
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sampling_params, use_tqdm=False)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        text = outs[0].outputs[0].text
        out_tokens.append(len(outs[0].outputs[0].token_ids))
        if save_predictions:
            raw_texts.append(text)
        parsed = parse_json_output(text)
        wide = {k: parsed.get(k, "") for k in fields}
        records.append(Record(fields=wide))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(prompts)}  "
                  f"p50={statistics.median(latencies):.0f}ms", flush=True)

    # Build wide gold records
    wide_gold = []
    for g, raw in zip(gold_12, raw_lines):
        wide = derive_fields(g.fields, raw, level)
        wide_gold.append(Record(fields={k: wide.get(k, "") for k in fields}))

    lat = summarize(latencies)
    f1_res = field_f1(records, wide_gold)
    quality = {
        "field_f1": round(f1_res["macro_field_f1"], 4),
        "schema_valid_pct": round(schema_validity(records), 2),
        "consistency_pct": round(consistency(records), 2),
        "n_fields": len(fields),
    }
    result = {
        "level": level,
        "n_fields": len(fields),
        "latency": lat,
        "quality": quality,
        "mean_output_tokens": float(statistics.fmean(out_tokens)),
        "tail_ratio": lat.get("p99_9_ms", 0) / lat.get("p50_ms", 1)
            if lat.get("p50_ms") else None,
    }
    if save_predictions and raw_texts:
        result["raw_texts_sample"] = raw_texts[:10]
        result["raw_token_counts"] = out_tokens
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--source", default="bgl",
                    choices=["hdfs", "bgl", "thunderbird", "spirit"])
    ap.add_argument("--levels", type=int, nargs="+",
                    default=[12, 25, 50, 100])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--denoising-steps", type=int, default=None)
    ap.add_argument("--output", default="results/wide_schema.json")
    ap.add_argument("--save-predictions", action="store_true",
                    help="save per-record raw text + token count to .preds.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    raw_lines = load_lines(str(SPLIT_DIR / f"{args.source}.raw"), args.n)
    gold_12 = load_gold(str(SPLIT_DIR / f"{args.source}.jsonl"), args.n)
    print(f"Loaded {len(raw_lines)} {args.source} events")

    llm_kwargs = dict(
        model=args.model, dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enable_prefix_caching=True)
    if args.denoising_steps is not None:
        llm_kwargs["diffusion_config"] = {
            "canvas_length": 256,
            "max_denoising_steps": args.denoising_steps}
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=0.0)

    tokenizer = llm.get_tokenizer()

    results = {
        "experiment": "issue19_wide_schema",
        "model": args.model,
        "source": args.source,
        "n_events": len(raw_lines),
        "levels": args.levels,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "per_level": {},
    }

    for level in args.levels:
        print(f"\n=== schema level L{level} ===", flush=True)
        level_max_tokens = min(args.max_new_tokens, level * 20)
        level_params = SamplingParams(
            max_tokens=level_max_tokens, temperature=0.0)
        prompts = []
        for line in raw_lines:
            msgs = build_wide_chat_messages(
                line, args.source, level, few_shot=True)
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        entry = run_level(llm, level_params, prompts, gold_12,
                          raw_lines, level,
                          save_predictions=args.save_predictions)
        results["per_level"][str(level)] = entry
        q = entry["quality"]
        print(f"  L{level} ({q['n_fields']}F): p50={entry['latency']['p50_ms']:.0f}ms  "
              f"F1={q['field_f1']:.3f}  valid={q['schema_valid_pct']:.1f}%  "
              f"cons={q['consistency_pct']:.1f}%", flush=True)

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
