#!/usr/bin/env python3
"""
Issue #24: Adaptive-effort experiment — per-bin latency and quality.

Selects a stratified subset of BGL events (routine/medium/novel by template
frequency), runs the model, and groups per-event latency + F1 by bin.

Two modes (run in the matching venv):
  --dinfer : LLaDA-8B under dInfer (threshold decoder, adaptive iterations)
  --vllm-ar: AR-3B under vLLM (fixed cost, baseline)
  --mdm-hf : LLaDA-8B under HF MDMModel (fixed K=8, non-adaptive DM baseline)

Usage:
    # dInfer (threshold decoder)
    HF_HUB_OFFLINE=1 .venv-dinfer/bin/python models/experiments/run_adaptive_effort.py \
        --dinfer --n-per-bin 200

    # AR-3B (vLLM)
    HF_HUB_OFFLINE=1 .venv-dg/bin/python models/experiments/run_adaptive_effort.py \
        --vllm-ar --model Qwen/Qwen2.5-3B-Instruct --n-per-bin 200

    # Fixed-K DM baseline (HF)
    HF_HUB_OFFLINE=1 .venv-dinfer/bin/python models/experiments/run_adaptive_effort.py \
        --mdm-hf --n-per-bin 200
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
from metrics.quality_metrics import field_f1, schema_validity, consistency, anomaly_f1
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
BINS_PATH = _REPO_ROOT / "results" / "difficulty_bins.json"

MODEL_PATH = ("/home/alexkay28/.cache/huggingface/hub/models--GSAI-ML--"
              "LLaDA-8B-Instruct/snapshots/"
              "08b83a6feb34df1a6011b80c3c00c7563e963b07/")
MASK_ID = 126336
EOS_ID = 126081


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


def select_stratified(bins_data, log_lines, gold_records, n_per_bin):
    selected_indices = []
    for bin_name in ["routine", "medium", "novel"]:
        indices = bins_data.get(f"{bin_name}_indices", [])
        selected = indices[:n_per_bin]
        selected_indices.extend([(idx, bin_name) for idx in selected])
    selected_indices.sort(key=lambda x: x[0])
    sel_lines = [log_lines[idx] for idx, _ in selected_indices]
    sel_gold = [gold_records[idx] for idx, _ in selected_indices]
    sel_bins = [b for _, b in selected_indices]
    return sel_lines, sel_gold, sel_bins


def run_dinfer(args, log_lines, gold_records, bins):
    import torch
    from transformers import AutoTokenizer, AutoConfig
    from dinfer.model import LLaDAModelLM
    from dinfer.decoding.serving import SamplingParams, init_generator

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, config=config).eval()
    model.init_h2e_module()
    model = model.to(device)
    print(f"  loaded dInfer ({sum(p.numel() for p in model.parameters())/1e9:.1f}B)")

    sp = SamplingParams(mask_id=MASK_ID, eos_id=EOS_ID, cache="",
                        cont_weight=0.0, threshold=args.threshold, temperature=0.0,
                        parallel_decoding="threshold", early_stop=True,
                        use_bd=False, enable_torch_compile=False)
    dllm = init_generator(model, sp, backend="vllm", max_length=4096)

    wu = tok("warmup", return_tensors="pt")["input_ids"].to(device)
    dllm.generate(wu, gen_length=args.gen_len, block_length=args.block_length)
    torch.cuda.synchronize()
    print("  warmup done")

    records, lats = [], []
    for i, line in enumerate(log_lines):
        msgs = build_chat_messages(line, source="bgl", few_shot=True)
        prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        input_ids = tok(prompt_text, return_tensors="pt", truncation=True,
                        max_length=2048)["input_ids"].to(device)
        plen = input_ids.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = dllm.generate(input_ids, gen_length=args.gen_len,
                           block_length=args.block_length)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        gen = out[0][plen:]
        text = tok.decode(gen, skip_special_tokens=True)
        rec = _dict_to_record(parse_json_output(text))
        records.append(rec)
        lats.append(elapsed)
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(log_lines)}] avg={sum(lats)/len(lats):.0f}ms",
                  flush=True)
    return records, lats


def run_vllm_ar(args, log_lines, gold_records, bins):
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=4096, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.0)

    prompts = []
    for line in log_lines:
        msgs = build_chat_messages(line, source="bgl", few_shot=True)
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    records, lats = [], []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        lats.append((time.perf_counter() - t0) * 1000)
        text = outs[0].outputs[0].text
        rec = _dict_to_record(parse_json_output(text))
        records.append(rec)
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(prompts)}] avg={sum(lats)/len(lats):.0f}ms",
                  flush=True)
    return records, lats


def run_mdm_hf(args, log_lines, gold_records, bins):
    from models.baselines.mdm_wrapper import MDMModel

    model = MDMModel(model_name="GSAI-ML/LLaDA-8B-Instruct",
                     k_steps=args.k_steps, max_new_tokens=256)
    model.load()
    result = model.extract(log_lines, source="bgl", show_progress=True)
    return result.records, result.latencies_ms


def eval_per_bin(records, gold_records, lats, bins):
    result = {}
    for bin_name in ["routine", "medium", "novel"]:
        idxs = [i for i, b in enumerate(bins) if b == bin_name]
        if not idxs:
            continue
        bin_recs = [records[i] for i in idxs]
        bin_gold = [gold_records[i] for i in idxs]
        bin_lats = [lats[i] for i in idxs]
        f1_res = field_f1(bin_recs, bin_gold)
        sv = schema_validity(bin_recs)
        cons = consistency(bin_recs)
        anom = anomaly_f1(bin_recs, bin_gold)
        lat = latency_percentiles(bin_lats)
        lat["mean_ms"] = statistics.fmean(bin_lats) if bin_lats else 0
        result[bin_name] = {
            "n": len(idxs),
            "field_f1": round(f1_res["macro_field_f1"], 4),
            "schema_valid_pct": round(sv, 2),
            "consistency_pct": round(cons, 2),
            "anomaly_f1": round(anom["anomaly_f1"], 4),
            "p50_ms": round(lat.get("p50_ms", 0), 2),
            "p99_ms": round(lat.get("p99_ms", 0), 2),
            "mean_ms": round(lat["mean_ms"], 2),
        }
    return result


def main():
    ap = argparse.ArgumentParser(description="Issue #24: adaptive-effort")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dinfer", action="store_true")
    g.add_argument("--vllm-ar", action="store_true")
    g.add_argument("--mdm-hf", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-per-bin", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_label = "dinfer" if args.dinfer else ("ar3b" if args.vllm_ar else "mdm_hf")
    out_path = args.out or f"results/adaptive_effort_{model_label}.json"

    with open(BINS_PATH) as f:
        bins_data = json.load(f)

    max_idx = max(max(bins_data.get("novel_indices", [0])),
                  max(bins_data.get("routine_indices", [0])),
                  max(bins_data.get("medium_indices", [0]))) + 1
    print(f"Loading {max_idx} events...", flush=True)
    log_lines = load_log_lines(SPLIT_DIR / "bgl.raw", max_idx)
    gold_records = load_records(SPLIT_DIR / "bgl.jsonl", max_idx)

    sel_lines, sel_gold, sel_bins = select_stratified(
        bins_data, log_lines, gold_records, args.n_per_bin)
    print(f"Selected {len(sel_lines)} events: "
          f"routine={sel_bins.count('routine')} "
          f"medium={sel_bins.count('medium')} "
          f"novel={sel_bins.count('novel')}")

    if args.dinfer:
        records, lats = run_dinfer(args, sel_lines, sel_gold, sel_bins)
    elif args.vllm_ar:
        records, lats = run_vllm_ar(args, sel_lines, sel_gold, sel_bins)
    elif args.mdm_hf:
        records, lats = run_mdm_hf(args, sel_lines, sel_gold, sel_bins)

    per_bin = eval_per_bin(records, sel_gold, lats, sel_bins)

    result = {
        "experiment": "issue24_adaptive_effort",
        "model_type": model_label,
        "model": args.model if args.vllm_ar else "LLaDA-8B-Instruct",
        "threshold": args.threshold if args.dinfer else None,
        "k_steps": args.k_steps if args.mdm_hf else None,
        "n_per_bin": args.n_per_bin,
        "per_bin": per_bin,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n=== Per-bin results ===")
    print(f"{'Bin':>8}  {'N':>4}  {'F1':>6}  {'p50ms':>7}  {'p99ms':>7}  {'mean':>7}")
    for b in ["routine", "medium", "novel"]:
        if b in per_bin:
            r = per_bin[b]
            print(f"{b:>8}  {r['n']:4d}  {r['field_f1']:.3f}  "
                  f"{r['p50_ms']:7.0f}  {r['p99_ms']:7.0f}  {r['mean_ms']:7.0f}")


if __name__ == "__main__":
    main()
