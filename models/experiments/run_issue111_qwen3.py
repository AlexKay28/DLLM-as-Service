#!/usr/bin/env python3
"""Issue #111 / #112: Qwen3-30B-A3B MoE-matched AR baseline (single cell).

This is the AR-30B quality + single-cell latency harness. It was re-measured
incorrectly in pass 7 (issue #112): the gold record file was fed to the model
as the input log line, ``enforce_eager`` disabled CUDA graphs, and the wrong
9-field subset was scored. All three are fixed here so the harness matches the
anchor regime (``run_anchor_remeasure.py``):

  - model input  = ``datasets/splits/test/bgl.raw``  (raw log lines)
  - gold labels  = ``datasets/splits/test/bgl.jsonl`` (paired by index)
  - no ``enforce_eager``; ``enable_prefix_caching=True`` (graphs on, like anchors)
  - serial measurement: one prompt per ``generate`` call, per-event timing
  - predictions via ``_dict_to_record(parse_json_output(...))`` (not
    ``Record(**kwargs)``, which raises ``TypeError`` and silently empties every
    record -- the root cause of the ``field_f1_9 == 1.0`` contradiction)
  - 9-field macro read from ``field_f1`` (drops event_type, affected_entity,
    remediation_hint)
  - provenance block via ``capture_provenance``

Issue #121: added ``--seed``, ``--dataset``, ``--save-predictions``,
``--output``, and per-event JSONL dump (matching the DG pattern in
``run_anchor_remeasure.py``). Also writes ``max_median_ratio`` and
``cv_pct`` to the summary JSON for tail-statistics parity with the DG
n=2000 artifact.

Pinned stack (must match the anchors): vLLM 0.25.1 / torch 2.11.0 / CUDA 13.0.
``capture_provenance`` records the *actual* installed versions, so a mismatch
is visible in the emitted artifact.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
exec(open(str(REPO / "scripts" / "vllm_patch.py")).read())

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from metrics.schema import Record
from metrics.quality_metrics import field_f1, schema_validity, consistency
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.provenance import capture_provenance

SPLIT_DIR = REPO / "datasets" / "splits" / "test"


def load_log_lines(path, n=None):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.rstrip("\n")
            if line:
                lines.append(line)
    return lines


def load_gold_records(path, n=None):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                records.append(Record(fields=json.loads(line)))
    return records


def run_experiment(n_events=300, max_tokens=256, seed=42, dataset="bgl",
                   save_predictions=None, output=None):
    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen3-30B-A3B"
    print(f"Loading {model_name}...", flush=True)
    llm = LLM(
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=4096,
        enable_prefix_caching=True, trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    raw_path = str(SPLIT_DIR / f"{dataset}.raw")
    gold_path = str(SPLIT_DIR / f"{dataset}.jsonl")
    log_lines = load_log_lines(raw_path, n=n_events)
    gold_records = load_gold_records(gold_path, n=n_events)
    assert len(log_lines) == len(gold_records), (
        f"raw/gold length mismatch: {len(log_lines)} vs {len(gold_records)}")
    print(f"Loaded {len(log_lines)} {dataset.upper()} events "
          f"({len(gold_records)} gold)", flush=True)

    # Resample log_lines with the given seed to get a different event subset
    # per seed (deterministic via numpy RNG).
    if seed != 42:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(log_lines), size=len(log_lines), replace=False)
        log_lines = [log_lines[i] for i in idx]
        gold_records = [gold_records[i] for i in idx]

    prompts = []
    for line in log_lines:
        messages = build_chat_messages(line, source=dataset, few_shot=True)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=0,
                        seed=seed)

    # Serial measurement (batch-1): one prompt per generate call, per-event
    # wall-clock -- the regime the anchors and every other AR arm use.
    latencies = []
    records = []
    token_counts = []
    print(f"\n=== Serial sweep: L={max_tokens}, n={len(prompts)} ===", flush=True)
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        gt0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - gt0) * 1000.0)
        text = outs[0].outputs[0].text
        token_counts.append(len(outs[0].outputs[0].token_ids))
        records.append(_dict_to_record(parse_json_output(text)))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)}  p50={statistics.median(latencies):.1f}ms",
                  flush=True)
    wall_time = time.time() - t0

    f1_res = field_f1(records, gold_records[:len(records)])
    lat = latency_percentiles(latencies)

    # Tail statistics (parity with DG n=2000 artifact)
    max_lat = max(latencies)
    p50_lat = lat["p50_ms"]
    cv_pct = round(
        (statistics.stdev(latencies) / statistics.fmean(latencies) * 100.0)
        if len(latencies) > 1 else 0.0, 2)
    max_median_ratio = round(max_lat / p50_lat, 2) if p50_lat > 0 else 0.0

    prov = capture_provenance(
        model_name=model_name, model_type="ar30b", backend="vllm",
        max_new_tokens=max_tokens, parser_policy="first_object_wins",
        scoring_fields="9",
    )

    result = {
        "experiment": "issue111_ar30b_quality",
        "model": model_name,
        "model_type": "ar30b",
        "corpus": dataset.upper(),
        "seed": seed,
        "n_events": len(records),
        "max_new_tokens": max_tokens,
        "min_tokens": 0,
        "temperature": 0.0,
        "serving": "serial (batch-1)",
        "wall_time_s": round(wall_time, 2),
        "per_event_ms": round(statistics.fmean(latencies), 1),
        "mean_output_tokens": round(float(np.mean(token_counts)), 1),
        "ramble_rate_pct": round(
            100.0 * sum(1 for t in token_counts if t > max_tokens * 0.9)
            / max(len(token_counts), 1), 1),
        "latency_ms": {
            "p50": round(lat["p50_ms"], 1),
            "p99": round(lat["p99_ms"], 1),
            "p99_9": round(lat.get("p99_9_ms", max_lat), 1),
            "mean": round(statistics.fmean(latencies), 1),
            "max": round(max_lat, 1),
            "min": round(min(latencies), 1),
        },
        "max_median_ratio": max_median_ratio,
        "cv_pct": cv_pct,
        "field_f1_9": round(f1_res["macro_field_f1_9"], 4),
        "field_f1_12": round(f1_res["macro_field_f1_12"], 4),
        "schema_valid_pct": round(schema_validity(records), 1),
        "consistency_pct": round(consistency(records), 1),
        "per_field_f1": {k: round(v, 4) for k, v in f1_res.items()
                         if k.startswith("f1_")},
        "provenance": prov.to_dict(),
    }

    # Per-event JSONL dump (matches DG pattern in run_anchor_remeasure.py)
    if save_predictions:
        pred_path = Path(save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            for rec, lat_val in zip(records, latencies):
                f.write(json.dumps(
                    {"record": rec.fields, "latency_ms": lat_val}) + "\n")
        print(f"Per-event predictions saved to {pred_path}")

    out_path = Path(output) if output else (
        REPO / "results" / f"issue111_qwen3_30b_a3b_{dataset}_s{seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")

    print(f"\n=== RESULT (L={max_tokens}) ===")
    print(f"  per_event (mean): {result['per_event_ms']:.1f}ms")
    print(f"  p50/p99:          {result['latency_ms']['p50']:.1f} / "
          f"{result['latency_ms']['p99']:.1f} ms")
    print(f"  mean_tokens:      {result['mean_output_tokens']:.1f}")
    print(f"  F1_9:             {result['field_f1_9']:.4f}")
    print(f"  schema_valid:     {result['schema_valid_pct']:.1f}%")
    print(f"  Compare AR-3B dense: 665.7ms, F1 0.852 (serial)")
    print(f"  Compare DG-26B-A4B:  378.3ms, F1 0.852 (serial)")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for event resampling (42 = no resample)")
    ap.add_argument("--dataset", choices=["bgl", "hdfs"], default="bgl")
    ap.add_argument("--save-predictions", type=str, default=None,
                    help="Path to save per-event JSONL "
                         "(record + latency_ms per line)")
    ap.add_argument("--output", type=str, default=None,
                    help="Path for summary JSON (default: "
                         "results/issue111_qwen3_30b_a3b_{dataset}_s{seed}.json)")
    args = ap.parse_args()
    run_experiment(n_events=args.n, max_tokens=args.max_tokens, seed=args.seed,
                   dataset=args.dataset, save_predictions=args.save_predictions,
                   output=args.output)
