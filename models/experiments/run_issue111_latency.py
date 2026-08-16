#!/usr/bin/env python3
"""Issue #111 / #112: serial cost-law sweep for Qwen3-30B-A3B (AR-30B).

Produces ``results/issue111_qwen3_serial_sweep.json`` -- the artifact the
paper's ``\\arBase``, ``\\arSlope`` and ``\\crossoverL`` macros trace to. It
was previously an orphan (no harness wrote this filename; a renamed copy of
the gold-leaked ``latency_sweep.json``), and it carried no provenance, so
``verify_numbers.py`` could not see it (issue #112, grill-pass7 finding 36).

Fixed to match the anchor regime (``run_anchor_remeasure.py``):

  - model input = ``bgl.raw``, gold = ``bgl.jsonl`` (paired by index)
  - no ``enforce_eager``; ``enable_prefix_caching=True``
  - serial (batch-1) per-event timing at each L -- not ``llm.generate(prompts)``
    batched throughput, which is what produced the implausible 48 ms/token
  - ``_dict_to_record`` + 9-field macro from ``field_f1``
  - provenance block + ramble rate (issue #112 Phase 2 step 9)

Pinned stack: vLLM 0.25.1 / torch 2.11.0 / CUDA 13.0 (must match anchors).
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
from metrics.quality_metrics import field_f1, schema_validity
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.provenance import capture_provenance

SPLIT_DIR = REPO / "datasets" / "splits" / "test"

# DG crossover constants (from the DG anchor, unaffected by this issue).
DG_BASE, DG_STEP = 378.8, 119.5


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


def _is_valid(text):
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and "severity" in obj and "host" in obj
    except (json.JSONDecodeError, ValueError):
        return False


def run_experiment(n_events=300, lengths=None):
    from vllm import LLM, SamplingParams

    if lengths is None:
        lengths = [32, 128, 256, 512, 1024]

    model_name = "Qwen/Qwen3-30B-A3B"
    print(f"Loading {model_name}...", flush=True)
    llm = LLM(
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=4096,
        enable_prefix_caching=True, trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    log_lines = load_log_lines(str(SPLIT_DIR / "bgl.raw"), n=n_events)
    gold_records = load_gold_records(str(SPLIT_DIR / "bgl.jsonl"), n=n_events)
    assert len(log_lines) == len(gold_records), (
        f"raw/gold length mismatch: {len(log_lines)} vs {len(gold_records)}")
    print(f"Loaded {len(log_lines)} BGL events ({len(gold_records)} gold)",
          flush=True)

    prompts = []
    for line in log_lines:
        messages = build_chat_messages(line, source="bgl", few_shot=True)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    results = {}
    for max_tokens in lengths:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=0)
        print(f"\n=== max_tokens={max_tokens} (serial, n={len(prompts)}) ===",
              flush=True)

        latencies = []
        records = []
        token_counts = []
        output_texts = []
        t0 = time.time()
        for i, prompt in enumerate(prompts):
            gt0 = time.perf_counter()
            outs = llm.generate([prompt], sp, use_tqdm=False)
            latencies.append((time.perf_counter() - gt0) * 1000.0)
            otext = outs[0].outputs[0].text
            output_texts.append(otext)
            token_counts.append(len(outs[0].outputs[0].token_ids))
            records.append(_dict_to_record(parse_json_output(otext)))
            if (i + 1) % 50 == 0:
                print(f"  L={max_tokens} {i+1}/{len(prompts)}  "
                      f"p50={statistics.median(latencies):.1f}ms", flush=True)
        wall_time = time.time() - t0

        f1_res = field_f1(records, gold_records[:len(records)])
        lat = latency_percentiles(latencies)
        valid = sum(1 for t in output_texts if _is_valid(t))
        ramble = sum(1 for tc in token_counts if tc >= max_tokens - 1)

        result = {
            "max_tokens": max_tokens,
            "n_events": len(records),
            "serving": "serial (batch-1)",
            "wall_time_s": round(wall_time, 2),
            "per_event_ms": round(statistics.fmean(latencies), 1),
            "mean_output_tokens": round(float(np.mean(token_counts)), 1),
            "ramble_rate_pct": round(100.0 * ramble / max(len(token_counts), 1), 1),
            "latency_ms": {
                "p50": round(lat["p50_ms"], 1),
                "p99": round(lat["p99_ms"], 1),
                "mean": round(statistics.fmean(latencies), 1),
            },
            "field_f1_9": round(f1_res["macro_field_f1_9"], 4),
            "field_f1_12": round(f1_res["macro_field_f1_12"], 4),
            "schema_valid_pct": round(100.0 * valid / max(len(output_texts), 1), 1),
            "per_field_f1": {k: round(v, 4) for k, v in f1_res.items()
                             if k.startswith("f1_")},
        }
        results[f"L{max_tokens}"] = result
        print(f"  per_event: {result['per_event_ms']:.1f}ms, "
              f"tokens: {result['mean_output_tokens']:.1f}, "
              f"F1_9: {result['field_f1_9']:.4f}, "
              f"valid: {result['schema_valid_pct']:.1f}%, "
              f"ramble: {result['ramble_rate_pct']:.1f}%")

    # Cost law: T = a + b * L  (L = mean output tokens actually produced)
    Ls = [results[f"L{mt}"]["mean_output_tokens"] for mt in lengths]
    Ts = [results[f"L{mt}"]["per_event_ms"] for mt in lengths]
    coeffs = np.polyfit(Ls, Ts, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    crossover = (DG_BASE - intercept) / slope if slope > 0 else None

    print(f"\n=== Cost law: T = {intercept:.1f} + {slope:.2f} * L ===")
    print(f"  Compare: AR-3B dense slope = 6.3 ms/token")
    if crossover is not None:
        print(f"  Crossover L* = ({DG_BASE} - {intercept:.1f}) / {slope:.2f} "
              f"= {crossover:.0f} tokens")

    results["cost_law"] = {
        "intercept_ms": round(intercept, 1),
        "slope_ms_per_token": round(slope, 2),
        "crossover_L_star": round(crossover, 0) if crossover is not None else None,
        "dg_base_ms": DG_BASE,
        "dg_step_ms": DG_STEP,
        "fit_L_values": Ls,
        "fit_T_values": Ts,
    }

    prov = capture_provenance(
        model_name=model_name, model_type="ar30b", backend="vllm",
        max_new_tokens=int(lengths[-1]), parser_policy="first_object_wins",
        scoring_fields="9",
    )
    results["experiment"] = "issue111_ar30b_serial_sweep"
    results["model"] = model_name
    results["model_type"] = "ar30b"
    results["corpus"] = "BGL"
    results["n_events"] = n_events
    results["serving"] = "serial (batch-1)"
    results["lengths"] = lengths
    results["provenance"] = prov.to_dict()

    out_path = REPO / "results" / "issue111_qwen3_serial_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {out_path}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300,
                    help="events per L (issue #112 raises this from 50)")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[32, 128, 256, 512, 1024])
    args = ap.parse_args()
    run_experiment(n_events=args.n, lengths=args.lengths)
