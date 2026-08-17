#!/usr/bin/env python3
"""Issue #119: Anytime K-cascade — check quality at K=2, escalate to K=8 only if needed.

Tests whether DG's latency can be made input-dependent: easy inputs finish at
K=2 (~200ms), hard inputs escalate to K=8 (~380ms). AR has no equivalent — it
cannot check quality mid-generation.

Approach:
  1. Run DG at K=2 with logprobs=1 (per-token confidence)
  2. Compute per-record mean confidence
  3. If confidence >= threshold τ → accept the K=2 output
  4. If confidence < τ → re-run the same input at K=8
  5. Measure: average K used, average latency, F1 vs fixed K=8

Usage:
    .venv/bin/python models/experiments/run_anytime_cascade.py --n 300
    .venv/bin/python models/experiments/run_anytime_cascade.py --n 300 --thresholds 0.5 0.7 0.8 0.9
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

from metrics.schema import Record
from metrics.quality_metrics import field_f1, schema_validity, consistency
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.provenance import capture_provenance

SPLIT_DIR = REPO / "datasets" / "splits" / "test"
MODEL_NAME = "google/diffusiongemma-26B-A4B-it"


def load_data(n):
    raw = []
    gold = []
    with open(SPLIT_DIR / "bgl.raw") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            raw.append(line.strip())
    with open(SPLIT_DIR / "bgl.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            gold.append(Record(fields=json.loads(line)))
    return raw, gold


def run_cascade(n=300, thresholds=None, k_low=2, k_high=8, canvas=256):
    from vllm import LLM, SamplingParams

    if thresholds is None:
        thresholds = [0.5, 0.7, 0.8, 0.9]

    raw_lines, gold_records = load_data(n)
    print(f"Loaded {len(raw_lines)} BGL events", flush=True)

    print(f"Loading {MODEL_NAME}...", flush=True)
    llm = LLM(
        model=MODEL_NAME, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=4096,
        enable_prefix_caching=True,
        diffusion_config={"canvas_length": canvas, "max_denoising_steps": k_high},
    )
    tok = llm.get_tokenizer()

    prompts = []
    for line in raw_lines:
        msgs = build_chat_messages(line, source="bgl", few_shot=True)
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    # Phase 1: Run all inputs at K_low with logprobs
    print(f"\n=== Phase 1: K={k_low} (low-cost pass) with logprobs ===", flush=True)
    sp_low = SamplingParams(temperature=0.0, max_tokens=canvas, min_tokens=0,
                            logprobs=1)
    # Override denoising steps for low pass
    sp_low.denoising_steps = k_low

    t0 = time.time()
    outputs_low = llm.generate(prompts, sp_low)
    wall_low = time.time() - t0

    low_results = []
    for out in outputs_low:
        text = out.outputs[0].text
        token_ids = out.outputs[0].token_ids
        # Compute mean confidence from logprobs
        confidences = []
        for logprob_entry in out.outputs[0].logprobs:
            if logprob_entry:
                # logprob_entry is {token_id: Logprob}
                best = max(logprob_entry.values(), key=lambda x: x.logprob)
                confidences.append(best.logprob)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        # Convert log prob to probability
        import math
        mean_prob = math.exp(mean_conf) if mean_conf < 0 else 1.0
        low_results.append({
            "text": text,
            "token_count": len(token_ids),
            "mean_confidence": mean_prob,
            "latency_ms": wall_low * 1000 / len(outputs_low),  # batched, per-event
        })

    confs = [r["mean_confidence"] for r in low_results]
    print(f"  K={k_low}: p50 conf={statistics.median(confs):.3f}, "
          f"mean={statistics.fmean(confs):.3f}", flush=True)

    # Phase 2: Run all inputs at K_high (for comparison baseline)
    print(f"\n=== Phase 2: K={k_high} (full-quality baseline) ===", flush=True)
    sp_high = SamplingParams(temperature=0.0, max_tokens=canvas, min_tokens=0)
    sp_high.denoising_steps = k_high

    t0 = time.time()
    outputs_high = llm.generate(prompts, sp_high)
    wall_high = time.time() - t0

    high_texts = [out.outputs[0].text for out in outputs_high]
    high_records = [_dict_to_record(parse_json_output(t)) for t in high_texts]
    f1_high = field_f1(high_records, gold_records[:len(high_records)])
    print(f"  K={k_high}: F1_9={f1_high['macro_field_f1_9']:.4f}", flush=True)

    # Phase 3: For each threshold, simulate the cascade
    print(f"\n=== Phase 3: Cascade analysis ===", flush=True)
    cascade_results = {}

    for tau in thresholds:
        # Inputs below threshold get re-run at K_high
        escalate_count = sum(1 for r in low_results if r["mean_confidence"] < tau)
        accept_count = n - escalate_count

        # Build cascade records: accepted from K_low, escalated from K_high
        cascade_records = []
        cascade_lats = []
        for i, (low_r, high_text) in enumerate(zip(low_results, high_texts)):
            if low_r["mean_confidence"] >= tau:
                # Accept K_low output
                rec = _dict_to_record(parse_json_output(low_r["text"]))
                cascade_records.append(rec)
                cascade_lats.append(low_r["latency_ms"])
            else:
                # Use K_high output
                rec = _dict_to_record(parse_json_output(high_text))
                cascade_records.append(rec)
                # Latency = K_low + K_high (sequential)
                cascade_lats.append(low_r["latency_ms"] + wall_high * 1000 / n)

        f1_cascade = field_f1(cascade_records, gold_records[:len(cascade_records)])
        avg_k = (accept_count * k_low + escalate_count * (k_low + k_high)) / n
        avg_lat = statistics.fmean(cascade_lats)

        result = {
            "threshold": tau,
            "accept_k_low": accept_count,
            "escalate_to_k_high": escalate_count,
            "accept_rate": round(accept_count / n, 4),
            "avg_k": round(avg_k, 2),
            "avg_latency_ms": round(avg_lat, 1),
            "p50_latency_ms": round(statistics.median(cascade_lats), 1),
            "field_f1_9": round(f1_cascade["macro_field_f1_9"], 4),
            "schema_valid_pct": round(schema_validity(cascade_records), 1),
        }
        cascade_results[f"tau_{tau}"] = result

        print(f"  τ={tau}: accept={accept_rate:.1%}  avg_K={avg_k:.1f}  "
              f"avg_lat={avg_lat:.0f}ms  F1={f1_cascade['macro_field_f1_9']:.4f}",
              flush=True)

    # Baseline comparison
    print(f"\n=== Baselines ===")
    print(f"  Fixed K={k_low}:  F1={field_f1([_dict_to_record(parse_json_output(r['text'])) for r in low_results], gold_records[:n])['macro_field_f1_9']:.4f}  "
          f"lat={low_results[0]['latency_ms']:.0f}ms")
    print(f"  Fixed K={k_high}: F1={f1_high['macro_field_f1_9']:.4f}  "
          f"lat={wall_high*1000/n:.0f}ms")

    prov = capture_provenance(
        model_name=MODEL_NAME, model_type="dg", backend="vllm",
        max_new_tokens=canvas, parser_policy="first_object_wins",
        scoring_fields="9",
    )
    prov.denoising_steps = k_high
    prov.canvas_length = canvas

    result = {
        "experiment": "issue119_anytime_cascade",
        "model": MODEL_NAME,
        "model_type": "dg",
        "n_events": n,
        "k_low": k_low,
        "k_high": k_high,
        "canvas_length": canvas,
        "serving": "cascade (K_low + conditional K_high)",
        "confidence_stats": {
            "p50": round(statistics.median(confs), 4),
            "mean": round(statistics.fmean(confs), 4),
            "min": round(min(confs), 4),
            "max": round(max(confs), 4),
        },
        "baselines": {
            f"fixed_K{k_low}": {
                "field_f1_9": round(field_f1(
                    [_dict_to_record(parse_json_output(r["text"])) for r in low_results],
                    gold_records[:n])["macro_field_f1_9"], 4),
                "avg_latency_ms": round(low_results[0]["latency_ms"], 1),
            },
            f"fixed_K{k_high}": {
                "field_f1_9": round(f1_high["macro_field_f1_9"], 4),
                "avg_latency_ms": round(wall_high * 1000 / n, 1),
            },
        },
        "cascade": cascade_results,
        "provenance": prov.to_dict(),
    }

    out_dir = REPO / "results" / "issue119_cascade"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cascade_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9])
    ap.add_argument("--k-low", type=int, default=2)
    ap.add_argument("--k-high", type=int, default=8)
    args = ap.parse_args()
    run_cascade(n=args.n, thresholds=args.thresholds,
                k_low=args.k_low, k_high=args.k_high)
