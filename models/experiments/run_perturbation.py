#!/usr/bin/env python3
"""
Issue #27: Perturbation / drift robustness — DM corrects input noise.

Applies each perturbation type from the existing suite at multiple
intensities, runs extraction, measures per-category F1 drop and
latency change vs clean baseline.

Usage:
    # AR-3B (vLLM)
    HF_HUB_OFFLINE=1 .venv-dg/bin/python models/experiments/run_perturbation.py \
        --vllm-ar --model Qwen/Qwen2.5-3B-Instruct --n 500

    # DM (HF MDMModel, K=8)
    HF_HUB_OFFLINE=1 .venv-dinfer/bin/python models/experiments/run_perturbation.py \
        --mdm-hf --n 500
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
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from models.perturbations.perturb import (
    PerturbationType, PerturbationConfig, perturb_logs,
)

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"

PERTURBATION_TYPES = [
    "WHITESPACE", "FIELD_REORDER", "SEPARATOR_CHANGE", "ABBREVIATION",
    "TIMESTAMP_FORMAT", "MISSING_FIELD", "CASE_VARIATION",
    "WRAPPING_NOISE", "SYNONYM_SUBSTITUTION", "ENCODING_NOISE",
]
INTENSITIES = [0.3, 0.6, 1.0]


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


def extract_ar(llm, tokenizer, sp, log_lines):
    from models.baselines.ar_model import build_chat_messages
    records, lats = [], []
    for line in log_lines:
        msgs = build_chat_messages(line, source="bgl", few_shot=True)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        lats.append((time.perf_counter() - t0) * 1000)
        rec = _dict_to_record(parse_json_output(outs[0].outputs[0].text))
        records.append(rec)
    return records, lats


def extract_mdm(model, log_lines):
    records, lats = [], []
    for line in log_lines:
        t0 = time.perf_counter()
        result = model._generate_single(line, "bgl")
        lats.append((time.perf_counter() - t0) * 1000)
        rec = _dict_to_record(parse_json_output(result[0]))
        records.append(rec)
    return records, lats


def eval_category(records, gold_records, lats):
    f1_res = field_f1(records, gold_records)
    sv = schema_validity(records)
    cons = consistency(records)
    lat = latency_percentiles(lats)
    out = {
        "field_f1": round(f1_res["macro_field_f1"], 4),
        "schema_valid_pct": round(sv, 2),
        "consistency_pct": round(cons, 2),
        "p50_ms": round(lat.get("p50_ms", 0), 2),
        "mean_ms": round(statistics.fmean(lats) if lats else 0, 2),
    }
    for f in FIELD_NAMES:
        out[f"f1_{f}"] = round(f1_res.get(f"f1_{f}", 0.0), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description="Issue #27: perturbation robustness")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vllm-ar", action="store_true")
    g.add_argument("--mdm-hf", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--intensities", nargs="+", type=float, default=INTENSITIES)
    ap.add_argument("--categories", nargs="+", default=None,
                    help="Subset of PERTURBATION_TYPES to run (default: all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_label = "ar3b" if args.vllm_ar else "dm_hf"
    out_path = args.out or f"results/perturbation_{model_label}.json"

    log_lines = load_log_lines(SPLIT_DIR / "bgl.raw", args.n)
    gold_records = load_records(SPLIT_DIR / "bgl.jsonl", args.n)
    print(f"Loaded {len(log_lines)} events")

    if args.vllm_ar:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=4096, enable_prefix_caching=True)
        tokenizer = llm.get_tokenizer()
        sp = SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.0)
        extract_fn = lambda lines: extract_ar(llm, tokenizer, sp, lines)
    else:
        from models.baselines.mdm_wrapper import MDMModel
        model = MDMModel(model_name="GSAI-ML/LLaDA-8B-Instruct",
                         k_steps=args.k_steps, max_new_tokens=256)
        model.load()
        extract_fn = lambda lines: extract_mdm(model, lines)

    print("\n  Clean baseline...", flush=True)
    records, lats = extract_fn(log_lines)
    baseline = eval_category(records, gold_records, lats)
    print(f"    F1={baseline['field_f1']:.3f} p50={baseline['p50_ms']:.0f}ms")

    categories = {"clean": baseline}
    ptype_names = args.categories if args.categories else PERTURBATION_TYPES
    for ptype_name in ptype_names:
        ptype = PerturbationType[ptype_name]
        for intensity in args.intensities:
            key = f"{ptype_name.lower()}_i{intensity:.1f}"
            print(f"\n  {ptype_name} @ {intensity}...", flush=True)
            config = PerturbationConfig(perturbation_type=ptype,
                                        intensity=intensity, seed=42)
            perturbed = perturb_logs(log_lines, config)
            records, lats = extract_fn(perturbed)
            cat = eval_category(records, gold_records, lats)
            cat["f1_drop"] = round(baseline["field_f1"] - cat["field_f1"], 4)
            cat["latency_ratio"] = round(cat["p50_ms"] / baseline["p50_ms"], 3)
            categories[key] = cat
            print(f"    F1={cat['field_f1']:.3f} (drop={cat['f1_drop']:.3f}) "
                  f"p50={cat['p50_ms']:.0f}ms (ratio={cat['latency_ratio']:.2f})")

    result = {
        "experiment": "issue27_perturbation_robustness",
        "model_type": model_label,
        "model": args.model,
        "n": len(log_lines),
        "categories": categories,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    print(f"\n{'Category':>30}  {'F1':>6}  {'Drop':>6}  {'p50ms':>7}  {'Ratio':>6}")
    for k, v in sorted(categories.items()):
        drop = v.get("f1_drop", 0)
        ratio = v.get("latency_ratio", 1.0)
        print(f"{k:>30}  {v['field_f1']:.3f}  {drop:.3f}  "
              f"{v['p50_ms']:7.0f}  {ratio:.2f}")


if __name__ == "__main__":
    main()
