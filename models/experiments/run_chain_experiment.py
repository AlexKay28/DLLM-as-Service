#!/usr/bin/env python3
"""
Issue #21 (App E): depth-first investigation chains on BGL.

Tests whether dLLM's per-call latency advantage (DG 420ms vs AR 807ms at
m=1) compounds with chain depth D, and whether the tail-ratio penalty
(DG 3.05x vs AR 2.01x at single hop) washes out under sequential
composition (CLT).

Each "chain" is D sequential single-request calls through vLLM at m=1
(one request in flight at a time).  Per-hop and per-chain cumulative
latencies are recorded.  The per-hop task is the standard 12-field BGL
extraction (natural ~150 output tokens, above the 65-token crossover).

Uniform-hop GPU run validates P-E1/E3/E4; the mixed-hop case (R1:
decision hops short, extraction hops long) is simulated offline by
analyze_chains.py from the per-hop marginals recorded here.

Usage:
    python models/experiments/run_chain_experiment.py \
        --model google/diffusiongemma-26B-A4B-it \
        --depths 1 3 5 10 20 --chains 100
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
from metrics.quality_metrics import field_f1, schema_validity
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"


def load_log_lines(path, n):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_records(path, n):
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
        out["min_ms"] = float(min(values))
        out["max_ms"] = float(max(values))
    out["n_samples"] = len(values)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--source", default="bgl",
                    choices=["hdfs", "bgl", "thunderbird", "spirit"])
    ap.add_argument("--depths", type=int, nargs="+",
                    default=[1, 3, 5, 10, 20],
                    help="chain depths to sweep")
    ap.add_argument("--chains", type=int, default=100,
                    help="number of chains per depth")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--denoising-steps", type=int, default=None)
    ap.add_argument("--output", default="results/chain_experiment.json")
    ap.add_argument("--save-predictions", action="store_true",
                    help="save per-hop raw text + parsed record to a .jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    total_events = args.chains * max(args.depths)
    log_lines = load_log_lines(str(SPLIT_DIR / f"{args.source}.raw"), n=total_events)
    gold_records = load_records(str(SPLIT_DIR / f"{args.source}.jsonl"),
                                n=total_events)
    print(f"Loaded {len(log_lines)} {args.source} events for "
          f"{args.chains} chains x depths {args.depths}")

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
    prompts = []
    for line in log_lines:
        messages = build_chat_messages(line, source=args.source, few_shot=True)
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    results = {
        "experiment": "issue21_chain_investigation",
        "model": args.model,
        "source": args.source,
        "chains_per_depth": args.chains,
        "depths": args.depths,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "per_depth": {},
        "per_hop_latencies_ms": [],
    }

    hop_records_all = []
    hop_raw_texts_all = []
    hop_lats_all = []
    depth_pred_ranges = []

    cursor = 0
    for D in args.depths:
        print(f"\n=== depth D={D} ===", flush=True)
        chain_cumul = []
        hop_lats = []
        hop_records = []
        hop_golds = []
        hop_out_tokens = []
        hop_raw_texts = []

        for c in range(args.chains):
            chain_lat = 0.0
            for h in range(D):
                prompt = prompts[cursor % len(prompts)]
                gold = gold_records[cursor % len(prompts)]
                cursor += 1
                t0 = time.perf_counter()
                outs = llm.generate([prompt], sampling_params, use_tqdm=False)
                dt = (time.perf_counter() - t0) * 1000.0
                chain_lat += dt
                hop_lats.append(dt)
                hop_out_tokens.append(
                    len(outs[0].outputs[0].token_ids))
                raw_text = outs[0].outputs[0].text
                hop_raw_texts.append(raw_text)
                hop_records.append(
                    _dict_to_record(parse_json_output(raw_text)))
                hop_golds.append(gold)
            chain_cumul.append(chain_lat)
            if (c + 1) % 20 == 0:
                print(f"    chain {c+1}/{args.chains}  "
                      f"cumul p50={statistics.median(chain_cumul):.0f}ms",
                      flush=True)

        cumul_summary = summarize(chain_cumul)
        hop_summary = summarize(hop_lats)
        f1_res = field_f1(hop_records, hop_golds)
        quality = {
            "field_f1": round(f1_res["macro_field_f1"], 4),
            "schema_valid_pct": round(schema_validity(hop_records), 2),
        }
        entry = {
            "depth": D,
            "cumulative_latency": cumul_summary,
            "per_hop_latency": hop_summary,
            "quality": quality,
            "mean_output_tokens": float(statistics.fmean(hop_out_tokens)),
            "chain_cumulative_latencies_ms": chain_cumul,
            "per_hop_latencies_ms_sample": hop_lats[:200],
        }
        results["per_depth"][str(D)] = entry
        results["per_hop_latencies_ms"].extend(hop_lats)
        depth_pred_ranges.append((D, len(hop_records_all),
                                   len(hop_records_all) + len(hop_records)))
        hop_records_all.extend(hop_records)
        hop_raw_texts_all.extend(hop_raw_texts)
        hop_lats_all.extend(hop_lats)
        print(f"  D={D}: cumul p50={cumul_summary['p50_ms']:.0f}ms  "
              f"p99={cumul_summary['p99_ms']:.0f}ms  "
              f"p99.9={cumul_summary['p99_9_ms']:.0f}ms  "
              f"per-hop p50={hop_summary['p50_ms']:.0f}ms  "
              f"F1={quality['field_f1']:.3f}", flush=True)

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    if args.save_predictions:
        preds_path = out_path.with_suffix(".preds.jsonl")
        with open(preds_path, "w") as f:
            for D, start, end in depth_pred_ranges:
                for i in range(start, end):
                    rec = hop_records_all[i]
                    f.write(json.dumps({
                        "depth": D,
                        "raw": hop_raw_texts_all[i],
                        "record": rec.fields,
                        "latency_ms": hop_lats_all[i],
                    }) + "\n")
        print(f"Wrote {preds_path}")


if __name__ == "__main__":
    main()
