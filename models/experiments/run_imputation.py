#!/usr/bin/env python3
"""
Issue #23: Missing-field imputation experiment.

Drops K fields from BGL records. AR (vLLM) regenerates the full JSON;
DM (HF MDMModel) selectively masks only the missing positions. Scores
F1 on the imputed subset only.

Usage:
    # Dry-run (no GPU)
    python models/experiments/run_imputation.py --mock --k 1 2 4 --n 50

    # AR-3B under vLLM
    python models/experiments/run_imputation.py --model Qwen/Qwen2.5-3B-Instruct \
        --k 1 2 4 6 8 --n 500

    # DM (HF MDMModel, LLaDA-8B)
    python models/experiments/run_imputation.py --mdm \
        --model GSAI-ML/LLaDA-8B-Instruct --k 1 2 4 6 8 --n 500
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, SCHEMA_FIELDS, FieldCategory
from metrics.quality_metrics import field_f1, schema_validity, consistency, anomaly_f1
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import parse_json_output
from models.baselines.ar_wrapper import _dict_to_record, ARModel, MockARModel

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"

STRUCTURAL_FIELDS = [f.name for f in SCHEMA_FIELDS if f.category == FieldCategory.STRUCTURAL]
SEMANTIC_FIELDS = [f.name for f in SCHEMA_FIELDS if f.category == FieldCategory.SEMANTIC]


def load_log_lines(path, n=None):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_records(path, n=None):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                records.append(Record(fields=json.loads(line)))
    return records


def sample_missing_fields(k: int, stratification: str, seed: int) -> list[str]:
    """Sample K field names to drop, same for all events in a cell."""
    rng = random.Random(seed)
    if stratification == "structural":
        candidates = STRUCTURAL_FIELDS
    elif stratification == "semantic":
        candidates = SEMANTIC_FIELDS
    else:
        candidates = FIELD_NAMES
    k = min(k, len(candidates))
    return rng.sample(candidates, k)


def make_known_missing(gold_records: list[Record], missing_fields: list[str]):
    """Split gold records into (known_fields, missing_fields) per event."""
    known_list = []
    missing_list = []
    for rec in gold_records:
        known = {f: rec.get(f) for f in FIELD_NAMES if f not in missing_fields}
        known_list.append(known)
        missing_list.append(list(missing_fields))
    return known_list, missing_list


def merge_known_predicted(known: dict, predicted: dict) -> Record:
    """Merge known fields with predicted fields; known takes priority."""
    merged = dict(known)
    for f in FIELD_NAMES:
        if f not in merged or merged[f] is None or merged[f] == "":
            merged[f] = predicted.get(f)
    return _dict_to_record(merged)


def run_ar_vllm(model_name, log_lines, known_list, missing_list, source, n):
    """Run AR under vLLM with imputation prompts (streaming m=1)."""
    from vllm import LLM, SamplingParams
    from models.baselines.imputation import build_ar_imputation_messages

    print(f"  loading vLLM: {model_name}...", flush=True)
    llm = LLM(model=model_name, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096,
              enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.0)

    prompts = []
    for line, known, missing in zip(log_lines, known_list, missing_list):
        msgs = build_ar_imputation_messages(line, source, known, missing)
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    records = []
    latencies = []
    print(f"  generating {len(prompts)} imputation calls (m=1)...", flush=True)
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        text = outs[0].outputs[0].text
        data = parse_json_output(text)
        rec = merge_known_predicted(known_list[i], data)
        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(prompts)}  p50={statistics.median(latencies):.1f}ms",
                  flush=True)

    return records, latencies


def run_dm_hf(model_name, log_lines, known_list, missing_list, source, k_steps):
    """Run DM under HF MDMModel with selective masking."""
    from models.baselines.mdm_wrapper import MDMModel
    print(f"  loading HF MDMModel: {model_name} (K={k_steps})...", flush=True)
    model = MDMModel(model_name=model_name, k_steps=k_steps, max_new_tokens=256)
    model.load()
    result = model.extract_impute(log_lines, known_list, missing_list,
                                  source=source, k_steps=k_steps)
    return result.records, result.latencies_ms


def run_mock(log_lines, known_list, missing_list, source, k_steps, is_mdm):
    """Mock dry-run."""
    if is_mdm:
        from models.baselines.mdm_wrapper import MockMDMModel
        model = MockMDMModel(k_steps=k_steps)
        result = model.extract_impute(log_lines, known_list, missing_list, source)
        return result.records, result.latencies_ms
    else:
        from models.baselines.drain_only import extract as drain_extract
        dr = drain_extract(log_lines, source=source)
        rng = random.Random(42)
        latencies = [rng.gauss(800, 100) for _ in log_lines]
        records = []
        for rec, known in zip(dr.records, known_list):
            records.append(merge_known_predicted(known, rec.fields))
        return records, latencies


def evaluate(records, gold_records, latencies, missing_fields):
    """Score on the imputed subset only."""
    f1_res = field_f1(records, gold_records, fields_to_eval=missing_fields)
    sv = schema_validity(records, skip_missing=missing_fields)
    cons = consistency(records)
    anom = anomaly_f1(records, gold_records)
    lat = latency_percentiles(latencies)
    lat["mean_ms"] = statistics.fmean(latencies) if latencies else 0
    return {
        "imputed_fields": missing_fields,
        "field_f1_imputed": f1_res["macro_field_f1"],
        "per_field_f1": {k: v for k, v in f1_res.items() if k.startswith("f1_")},
        "field_f1_all": f1_res.get("macro_field_f1"),
        "schema_validity": sv,
        "consistency": cons,
        "anomaly_f1": anom["anomaly_f1"],
        "latency": lat,
    }


def main():
    ap = argparse.ArgumentParser(description="Issue #23: imputation experiment")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--source", default="bgl")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    ap.add_argument("--stratification", default="random",
                    choices=["random", "structural", "semantic"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--mdm", action="store_true", help="use DM (HF MDMModel)")
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--mock", action="store_true", help="dry-run, no GPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw_path = SPLIT_DIR / f"{args.source}.raw"
    gold_path = SPLIT_DIR / f"{args.source}.jsonl"
    log_lines = load_log_lines(raw_path, args.n)
    gold_records = load_records(gold_path, args.n)
    print(f"  loaded {len(log_lines)} events from {args.source}")

    model_label = "DM" if args.mdm else "AR"
    out_path = args.out or f"results/imputation_{model_label}_{args.stratification}.json"

    all_results = {
        "experiment": "issue23_imputation",
        "model": args.model,
        "model_type": model_label,
        "source": args.source,
        "stratification": args.stratification,
        "n": len(log_lines),
        "k_steps": args.k_steps if args.mdm else None,
        "mock": args.mock,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": [],
    }

    for k_val in args.k:
        seed = args.seed + k_val
        missing = sample_missing_fields(k_val, args.stratification, seed)
        print(f"\n=== K={k_val}, stratification={args.stratification}, "
              f"missing={missing} ===", flush=True)

        known_list, missing_list = make_known_missing(gold_records, missing)

        if args.mock:
            records, latencies = run_mock(
                log_lines, known_list, missing_list, args.source,
                args.k_steps, args.mdm)
        elif args.mdm:
            records, latencies = run_dm_hf(
                args.model, log_lines, known_list, missing_list,
                args.source, args.k_steps)
        else:
            records, latencies = run_ar_vllm(
                args.model, log_lines, known_list, missing_list,
                args.source, args.n)

        cell = evaluate(records, gold_records, latencies, missing)
        cell["k"] = k_val
        all_results["cells"].append(cell)

        print(f"  F1(imputed)={cell['field_f1_imputed']:.3f}  "
              f"validity={cell['schema_validity']:.1f}%  "
              f"consistency={cell['consistency']:.1f}%  "
              f"p50={cell['latency'].get('p50_ms', 0):.1f}ms", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  wrote {out_path}", flush=True)

    print("\n=== Summary ===")
    print(f"{'K':>3}  {'F1(imp)':>8}  {'Validity':>8}  {'Consist':>8}  {'p50(ms)':>8}")
    for c in all_results["cells"]:
        print(f"{c['k']:3d}  {c['field_f1_imputed']:8.3f}  "
              f"{c['schema_validity']:7.1f}%  {c['consistency']:7.1f}%  "
              f"{c['latency'].get('p50_ms', 0):8.1f}")


if __name__ == "__main__":
    main()
