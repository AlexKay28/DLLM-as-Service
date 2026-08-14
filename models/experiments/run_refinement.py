#!/usr/bin/env python3
"""
Issue #24: Iterative refinement loops — DM re-masks only wrong fields.

Round 1: full extraction. Round 2+: critic identifies wrong fields,
DM re-masks only those (selective masking from #26), AR regenerates
full JSON. Tests whether DM's targeted refinement beats AR's full
regeneration in both quality gain and latency cost.

Usage:
    # DM refinement (HF MDMModel, selective re-masking)
    HF_HUB_OFFLINE=1 .venv-dinfer/bin/python models/experiments/run_refinement.py \
        --mdm-hf --rounds 1 2 3 --n 500

    # AR refinement (vLLM, full regeneration each round)
    HF_HUB_OFFLINE=1 .venv-dg/bin/python models/experiments/run_refinement.py \
        --vllm-ar --model Qwen/Qwen2.5-3B-Instruct --rounds 1 2 3 --n 500
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
from metrics.quality_metrics import field_f1, schema_validity, consistency, anomaly_f1
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.imputation import build_dm_imputation_prompt_text

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"


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


def critic_fields(predicted_rec, gold_rec):
    """Identify fields where prediction disagrees with gold."""
    wrong = []
    for f in FIELD_NAMES:
        pred_val = predicted_rec.get(f) if predicted_rec.has(f) else None
        gold_val = gold_rec.get(f) if gold_rec.has(f) else None
        if str(pred_val).strip().lower() != str(gold_val).strip().lower():
            wrong.append(f)
    return wrong


def run_ar_vllm_round(model_name, log_lines, prompts, sp, llm):
    """Run one AR extraction round via vLLM."""
    records, lats = [], []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        lats.append((time.perf_counter() - t0) * 1000)
        text = outs[0].outputs[0].text
        rec = _dict_to_record(parse_json_output(text))
        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(prompts)} avg={sum(lats)/len(lats):.0f}ms",
                  flush=True)
    return records, lats


def run_mdm_hf_round(model, log_lines, source):
    """Run one DM extraction round via HF MDMModel."""
    records, lats = [], []
    for i, line in enumerate(log_lines):
        t0 = time.perf_counter()
        result = model._generate_single(line, source)
        elapsed = (time.perf_counter() - t0) * 1000
        rec = _dict_to_record(parse_json_output(result[0]))
        records.append(rec)
        lats.append(elapsed)
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(log_lines)} avg={sum(lats)/len(lats):.0f}ms",
                  flush=True)
    return records, lats


def run_mdm_hf_refine(model, log_lines, prev_records, wrong_fields_list, source):
    """Run one DM refinement round via selective masking (extract_impute)."""
    records, lats = list(prev_records), [0.0] * len(prev_records)
    n_refined = 0
    for i, (line, wrong) in enumerate(zip(log_lines, wrong_fields_list)):
        if not wrong:
            continue
        correct_fields = [f for f in FIELD_NAMES if f not in wrong]
        known_fields = {}
        for f in correct_fields:
            if prev_records[i].has(f):
                known_fields[f] = prev_records[i].get(f)
        t0 = time.perf_counter()
        impute_result = model.extract_impute(
            log_lines=[line], known_fields_list=[known_fields],
            missing_fields_list=[wrong], source=source,
            show_progress=False, k_steps=model.k_steps,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        records[i] = impute_result.records[0]
        lats[i] = elapsed
        n_refined += 1
        if n_refined % 25 == 0:
            print(f"      refined {n_refined}/{sum(1 for w in wrong_fields_list if w)} "
                  f"avg={sum(l for l in lats if l > 0)/max(1,n_refined):.0f}ms",
                  flush=True)
    return records, lats, n_refined


def eval_round(records, gold_records, lats):
    f1_res = field_f1(records, gold_records)
    sv = schema_validity(records)
    cons = consistency(records)
    anom = anomaly_f1(records, gold_records)
    lat = latency_percentiles(lats)
    return {
        "field_f1": round(f1_res["macro_field_f1"], 4),
        "schema_valid_pct": round(sv, 2),
        "consistency_pct": round(cons, 2),
        "anomaly_f1": round(anom["anomaly_f1"], 4),
        "p50_ms": round(lat.get("p50_ms", 0), 2),
        "p99_ms": round(lat.get("p99_ms", 0), 2),
        "mean_ms": round(statistics.fmean(lats) if lats else 0, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Issue #24: iterative refinement")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vllm-ar", action="store_true")
    g.add_argument("--mdm-hf", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--source", default="bgl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_label = "ar3b" if args.vllm_ar else "dm_hf"
    out_path = args.out or f"results/refinement_{model_label}.json"

    log_lines = load_log_lines(SPLIT_DIR / f"{args.source}.raw", args.n)
    gold_records = load_records(SPLIT_DIR / f"{args.source}.jsonl", args.n)
    print(f"Loaded {len(log_lines)} events")

    all_rounds = {}
    prev_records = None

    max_round = max(args.rounds)

    if args.vllm_ar:
        from vllm import LLM, SamplingParams as VLLMSP
        llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=4096, enable_prefix_caching=True)
        tokenizer = llm.get_tokenizer()
        sp = VLLMSP(max_tokens=256, temperature=0.0, repetition_penalty=1.0)
        prompts = []
        for line in log_lines:
            msgs = build_chat_messages(line, source=args.source, few_shot=True)
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

        for r in range(1, max_round + 1):
            print(f"\n  Round {r}...", flush=True)
            if r == 1 or prev_records is None:
                records, lats = run_ar_vllm_round(args.model, log_lines, prompts, sp, llm)
            else:
                wrong_fields_list = [critic_fields(prev_records[i], gold_records[i])
                                    for i in range(len(prev_records))]
                n_wrong = sum(1 for w in wrong_fields_list if w)
                print(f"    critic: {n_wrong}/{len(prev_records)} need refinement",
                      flush=True)
                records, lats = run_ar_vllm_round(args.model, log_lines, prompts, sp, llm)

            round_result = eval_round(records, gold_records, lats)
            round_result["round"] = r
            if r in args.rounds:
                all_rounds[f"round_{r}"] = round_result
            print(f"    F1={round_result['field_f1']:.3f} p50={round_result['p50_ms']:.0f}ms")
            prev_records = records

    elif args.mdm_hf:
        from models.baselines.mdm_wrapper import MDMModel
        model = MDMModel(model_name="GSAI-ML/LLaDA-8B-Instruct",
                         k_steps=args.k_steps, max_new_tokens=256)
        model.load()

        for r in range(1, max_round + 1):
            print(f"\n  Round {r}...", flush=True)
            if r == 1 or prev_records is None:
                records, lats = run_mdm_hf_round(model, log_lines, args.source)
            else:
                wrong_fields_list = [critic_fields(prev_records[i], gold_records[i])
                                    for i in range(len(prev_records))]
                n_wrong = sum(1 for w in wrong_fields_list if w)
                print(f"    critic: {n_wrong}/{len(prev_records)} need refinement",
                      flush=True)
                records, lats, n_refined = run_mdm_hf_refine(
                    model, log_lines, prev_records, wrong_fields_list, args.source)
                print(f"    refined {n_refined} events", flush=True)

            round_result = eval_round(records, gold_records, lats)
            round_result["round"] = r
            if r in args.rounds:
                all_rounds[f"round_{r}"] = round_result
            print(f"    F1={round_result['field_f1']:.3f} p50={round_result['p50_ms']:.0f}ms")
            prev_records = records

    result = {
        "experiment": "issue24_iterative_refinement",
        "model_type": model_label,
        "model": args.model,
        "n": len(log_lines),
        "rounds": all_rounds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    print(f"\n{'Round':>6}  {'F1':>6}  {'p50ms':>7}  {'p99ms':>7}")
    for k, r in sorted(all_rounds.items()):
        print(f"{k:>6}  {r['field_f1']:.3f}  {r['p50_ms']:7.0f}  {r['p99_ms']:7.0f}")


if __name__ == "__main__":
    main()
