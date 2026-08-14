#!/usr/bin/env python3
"""
Issue #26: Bounded-deadline completion — DM completes, AR truncates.

Sweeps a deadline D. AR gets max_tokens=floor(D/6.28) (what it can
generate in time); DM always generates one full canvas (completes).
Scores record completeness + quality.

Usage:
    # AR-3B (vLLM)
    HF_HUB_OFFLINE=1 .venv-dg/bin/python models/experiments/run_bounded_deadline.py \
        --vllm-ar --model Qwen/Qwen2.5-3B-Instruct --n 500

    # DM (HF MDMModel, K=8, one canvas)
    HF_HUB_OFFLINE=1 .venv-dinfer/bin/python models/experiments/run_bounded_deadline.py \
        --mdm-hf --n 500
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES
from metrics.quality_metrics import field_f1, consistency
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
AR_MS_PER_TOKEN = 6.28


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


def record_completeness(records):
    n_complete = sum(
        1 for rec in records
        if all(rec.has(f) for f in FIELD_NAMES)
    )
    return n_complete / len(records) * 100 if records else 0


def field_completeness(records):
    total = sum(
        sum(1 for f in FIELD_NAMES if rec.has(f))
        for rec in records
    )
    return total / len(records) / len(FIELD_NAMES) * 100 if records else 0


def run_ar_vllm(model_name, log_lines, gold_records, deadlines, source, n):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=4096, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()

    prompts = []
    for line in log_lines:
        msgs = build_chat_messages(line, source=source, few_shot=True)
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    all_cells = []
    for D in deadlines:
        max_tok = max(1, int(D / AR_MS_PER_TOKEN))
        sp = SamplingParams(max_tokens=max_tok, temperature=0.0, repetition_penalty=1.0)
        print(f"\n  D={D}ms (max_tokens={max_tok})...", flush=True)

        records, lats = [], []
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            outs = llm.generate([prompt], sp, use_tqdm=False)
            lats.append((time.perf_counter() - t0) * 1000)
            text = outs[0].outputs[0].text
            rec = _dict_to_record(parse_json_output(text))
            records.append(rec)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(prompts)}", flush=True)

        f1_res = field_f1(records, gold_records)
        rc = record_completeness(records)
        fc = field_completeness(records)
        lat = latency_percentiles(lats)
        cell = {
            "deadline_ms": D, "max_tokens": max_tok,
            "field_f1": round(f1_res["macro_field_f1"], 4),
            "record_completeness_pct": round(rc, 2),
            "field_completeness_pct": round(fc, 2),
            "p50_ms": round(lat.get("p50_ms", 0), 2),
            "mean_ms": round(statistics.fmean(lats), 2),
        }
        all_cells.append(cell)
        print(f"    F1={cell['field_f1']:.3f} complete={rc:.1f}% "
              f"field_complete={fc:.1f}% p50={cell['p50_ms']:.0f}ms")
    return all_cells


def run_mdm_hf(log_lines, gold_records, deadlines, source, k_steps, n):
    from models.baselines.mdm_wrapper import MDMModel

    model = MDMModel(model_name="GSAI-ML/LLaDA-8B-Instruct",
                     k_steps=k_steps, max_new_tokens=256)
    model.load()

    print(f"\n  DM (K={k_steps}, one canvas, ~418ms)...", flush=True)
    records, lats = [], []
    for i, line in enumerate(log_lines):
        t0 = time.perf_counter()
        result = model._generate_single(line, source)
        elapsed = (time.perf_counter() - t0) * 1000
        output_text = result[0]
        rec = _dict_to_record(parse_json_output(output_text))
        records.append(rec)
        lats.append(elapsed)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(log_lines)} avg={sum(lats)/len(lats):.0f}ms",
                  flush=True)

    f1_res = field_f1(records, gold_records)
    rc = record_completeness(records)
    fc = field_completeness(records)
    lat = latency_percentiles(lats)
    cell = {
        "deadline_ms": "none (one canvas)",
        "field_f1": round(f1_res["macro_field_f1"], 4),
        "record_completeness_pct": round(rc, 2),
        "field_completeness_pct": round(fc, 2),
        "p50_ms": round(lat.get("p50_ms", 0), 2),
        "mean_ms": round(statistics.fmean(lats), 2),
    }
    return [cell]


def main():
    ap = argparse.ArgumentParser(description="Issue #26: bounded-deadline")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vllm-ar", action="store_true")
    g.add_argument("--mdm-hf", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--source", default="bgl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--deadlines", nargs="+", type=int,
                    default=[200, 400, 600, 800, 1000, 1500, 2000])
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_label = "ar3b" if args.vllm_ar else "dm_hf"
    out_path = args.out or f"results/bounded_deadline_{model_label}.json"

    log_lines = load_log_lines(SPLIT_DIR / f"{args.source}.raw", args.n)
    gold_records = load_records(SPLIT_DIR / f"{args.source}.jsonl", args.n)
    print(f"Loaded {len(log_lines)} events")

    if args.vllm_ar:
        cells = run_ar_vllm(args.model, log_lines, gold_records,
                           args.deadlines, args.source, args.n)
    else:
        cells = run_mdm_hf(log_lines, gold_records, args.deadlines,
                          args.source, args.k_steps, args.n)

    result = {
        "experiment": "issue26_bounded_deadline",
        "model_type": model_label,
        "model": args.model,
        "n": len(log_lines),
        "cells": cells,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    print(f"\n{'D(ms)':>7}  {'F1':>6}  {'Record%':>8}  {'Field%':>8}  {'p50':>7}")
    for c in cells:
        print(f"{str(c['deadline_ms']):>7}  {c['field_f1']:.3f}  "
              f"{c['record_completeness_pct']:7.1f}%  "
              f"{c['field_completeness_pct']:7.1f}%  "
              f"{c['p50_ms']:7.0f}")


if __name__ == "__main__":
    main()
