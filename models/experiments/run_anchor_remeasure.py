#!/usr/bin/env python3
"""
#57 R2: Re-measure the four anchor cells >=5 times on the pinned stack.

Anchors:
  AR-3B  at L=256 and L=512  (Qwen/Qwen2.5-3B-Instruct)
  DG     at L=256 and L=512  (google/diffusiongemma-26B-A4B-it, K=8)

Each cell is run 5 times with n=300 events, saving per-event latencies and
predictions. Reports median, IQR, and CV across repeats.

Usage:
    # AR-3B anchors (uses .venv with vLLM 0.7.2)
    python models/experiments/run_anchor_remeasure.py --arm ar3b

    # DG anchors (uses .venv-dg with vLLM 0.25.1 + DG support)
    python models/experiments/run_anchor_remeasure.py --arm dg

    # Both
    python models/experiments/run_anchor_remeasure.py --arm both

    # Dry run (verify pipeline)
    python models/experiments/run_anchor_remeasure.py --arm ar3b --repeats 1 --n 50
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

from metrics.schema import Record
from metrics.quality_metrics import field_f1, schema_validity, consistency
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.provenance import capture_provenance

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
OUT_DIR = _REPO_ROOT / "results" / "anchors"


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


def run_ar_anchor(model_name, log_lines, gold_records, L, repeat, n):
    """Run AR-3B anchor cell: one repeat at output length L."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=L, min_tokens=L, temperature=0.0,
                        repetition_penalty=1.0)

    prompts = []
    for line in log_lines:
        messages = build_chat_messages(line, source="bgl", few_shot=True)
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    latencies = []
    records = []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        records.append(_dict_to_record(parse_json_output(outs[0].outputs[0].text)))
        if (i + 1) % 100 == 0:
            print(f"    r{repeat} L={L}: {i+1}/{len(prompts)}  "
                  f"p50={statistics.median(latencies):.1f}ms", flush=True)

    prov = capture_provenance(
        model_name=model_name, model_type="ar3b", backend="vllm",
        max_new_tokens=L, parser_policy="first_object_wins", scoring_fields="9",
    )

    # Save predictions
    pred_path = OUT_DIR / f"ar3b_L{L}_r{repeat}_preds.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w") as f:
        for rec, lat in zip(records, latencies):
            f.write(json.dumps({"record": rec.fields, "latency_ms": lat}) + "\n")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return latencies, records, prov


def run_dg_anchor(model_name, log_lines, gold_records, L, repeat, n, k_steps=8):
    """Run DiffusionGemma anchor cell: one repeat at output length L, K=8."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enable_prefix_caching=True,
        diffusion_config={"canvas_length": 256, "max_denoising_steps": k_steps},
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=L, min_tokens=L, temperature=0.0)

    prompts = []
    for line in log_lines:
        messages = build_chat_messages(line, source="bgl", few_shot=True)
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    latencies = []
    records = []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        records.append(_dict_to_record(parse_json_output(outs[0].outputs[0].text)))
        if (i + 1) % 100 == 0:
            print(f"    r{repeat} L={L}: {i+1}/{len(prompts)}  "
                  f"p50={statistics.median(latencies):.1f}ms", flush=True)

    prov = capture_provenance(
        model_name=model_name, model_type="dg", backend="vllm",
        max_new_tokens=L, parser_policy="first_object_wins", scoring_fields="9",
    )
    prov.denoising_steps = k_steps

    pred_path = OUT_DIR / f"dg_L{L}_r{repeat}_preds.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w") as f:
        for rec, lat in zip(records, latencies):
            f.write(json.dumps({"record": rec.fields, "latency_ms": lat}) + "\n")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return latencies, records, prov


def summarize_repeats(repeat_results, label):
    """Summarize across repeats: median, IQR, CV of p50."""
    p50s = [r["p50_ms"] for r in repeat_results]
    p99s = [r["p99_ms"] for r in repeat_results]
    f1s = [r["field_f1_9"] for r in repeat_results]

    summary = {
        "label": label,
        "n_repeats": len(repeat_results),
        "p50_ms": {
            "median": round(statistics.median(p50s), 2),
            "mean": round(statistics.fmean(p50s), 2),
            "min": round(min(p50s), 2),
            "max": round(max(p50s), 2),
            "iqr": round(_iqr(p50s), 2),
            "cv_pct": round(_cv(p50s), 2),
        },
        "p99_ms": {
            "median": round(statistics.median(p99s), 2),
            "mean": round(statistics.fmean(p99s), 2),
            "iqr": round(_iqr(p99s), 2),
            "cv_pct": round(_cv(p99s), 2),
        },
        "field_f1_9": {
            "median": round(statistics.median(f1s), 4),
            "mean": round(statistics.fmean(f1s), 4),
            "min": round(min(f1s), 4),
            "max": round(max(f1s), 4),
        },
        "repeats": repeat_results,
    }

    med = summary["p50_ms"]["median"]
    cv = summary["p50_ms"]["cv_pct"]
    status = "PASS" if cv < 2.0 else "INVESTIGATE"
    print(f"\n  {label}: p50 median={med:.1f}ms  CV={cv:.2f}%  [{status}]")
    print(f"    p50 range: {min(p50s):.1f} - {max(p50s):.1f} ms")
    print(f"    F1 (9-field): {min(f1s):.4f} - {max(f1s):.4f}")

    return summary


def _iqr(values):
    if len(values) < 2:
        return 0.0
    s = sorted(values)
    q1 = s[len(s) // 4]
    q3 = s[3 * len(s) // 4]
    return q3 - q1


def _cv(values):
    if len(values) < 2:
        return 0.0
    m = statistics.fmean(values)
    if m == 0:
        return 0.0
    return statistics.stdev(values) / m * 100


def main():
    ap = argparse.ArgumentParser(description="#57 R2: anchor re-measurement")
    ap.add_argument("--arm", choices=["ar3b", "dg", "both"], default="both")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--k-steps", type=int, default=8,
                    help="Denoising steps for DG (default 8)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_lines = load_log_lines(str(SPLIT_DIR / "bgl.raw"), n=args.n)
    gold_records = load_records(str(SPLIT_DIR / "bgl.jsonl"), n=args.n)
    print(f"Loaded {len(log_lines)} BGL events ({len(gold_records)} gold)")

    all_summaries = {}
    provenance = {}

    if args.arm in ("ar3b", "both"):
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        for L in args.lengths:
            label = f"ar3b_L{L}"
            repeat_results = []
            for r in range(1, args.repeats + 1):
                print(f"\n=== {label} repeat {r}/{args.repeats} ===", flush=True)
                lats, recs, prov = run_ar_anchor(
                    model_name, log_lines, gold_records, L, r, args.n)
                lat = latency_percentiles(lats)
                f1_res = field_f1(recs, gold_records[:len(recs)])
                repeat_results.append({
                    "repeat": r,
                    "p50_ms": round(lat["p50_ms"], 2),
                    "p99_ms": round(lat["p99_ms"], 2),
                    "mean_ms": round(statistics.fmean(lats), 2),
                    "field_f1_9": round(f1_res["macro_field_f1_9"], 4),
                    "field_f1_12": round(f1_res["macro_field_f1_12"], 4),
                    "schema_valid_pct": round(schema_validity(recs), 2),
                    "consistency_pct": round(consistency(recs), 2),
                    "n_predictions": len(recs),
                })
                provenance[label] = prov.to_dict()
            all_summaries[label] = summarize_repeats(repeat_results, label)

    if args.arm in ("dg", "both"):
        model_name = "google/diffusiongemma-26B-A4B-it"
        for L in args.lengths:
            label = f"dg_L{L}"
            repeat_results = []
            for r in range(1, args.repeats + 1):
                print(f"\n=== {label} repeat {r}/{args.repeats} ===", flush=True)
                lats, recs, prov = run_dg_anchor(
                    model_name, log_lines, gold_records, L, r, args.n,
                    k_steps=args.k_steps)
                lat = latency_percentiles(lats)
                f1_res = field_f1(recs, gold_records[:len(recs)])
                repeat_results.append({
                    "repeat": r,
                    "p50_ms": round(lat["p50_ms"], 2),
                    "p99_ms": round(lat["p99_ms"], 2),
                    "mean_ms": round(statistics.fmean(lats), 2),
                    "field_f1_9": round(f1_res["macro_field_f1_9"], 4),
                    "field_f1_12": round(f1_res["macro_field_f1_12"], 4),
                    "schema_valid_pct": round(schema_validity(recs), 2),
                    "consistency_pct": round(consistency(recs), 2),
                    "n_predictions": len(recs),
                })
                provenance[label] = prov.to_dict()
            all_summaries[label] = summarize_repeats(repeat_results, label)

    # Save combined results
    result = {
        "experiment": "issue57_anchor_remeasurement",
        "pinned_stack": {
            "vllm_version": provenance.get("ar3b_L256", {}).get("vllm_version"),
            "torch_version": provenance.get("ar3b_L256", {}).get("torch_version"),
            "gpu_model": provenance.get("ar3b_L256", {}).get("gpu_model"),
        },
        "n_events": args.n,
        "repeats": args.repeats,
        "k_steps": args.k_steps,
        "summaries": all_summaries,
        "provenance": provenance,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = OUT_DIR / "anchor_remeasure.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    # Final verdict
    print("\n=== Anchor Spread Summary ===")
    all_pass = True
    for label, s in all_summaries.items():
        cv = s["p50_ms"]["cv_pct"]
        status = "PASS (<2%)" if cv < 2.0 else "INVESTIGATE (>=2%)"
        if cv >= 2.0:
            all_pass = False
        print(f"  {label:12s}: p50={s['p50_ms']['median']:.1f}ms  "
              f"CV={cv:.2f}%  {status}")

    if all_pass:
        print("\n  ALL ANCHORS PASS: run-to-run spread < 2%")
    else:
        print("\n  SOME ANCHORS EXCEED 2% CV — investigate variance source")


if __name__ == "__main__":
    main()
