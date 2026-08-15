#!/usr/bin/env python3
"""Issue #42 — does disabling KV-cache reuse remove AR's batching headroom?

The paper claims dLLM's ~1x batch multiplier (vs AR's 77x) is caused by
bidirectional attention invalidating KV-cache reuse. This script tests
the causal link directly:

  1. AR with prefix caching ON  (default) → should have high multiplier
  2. AR with prefix caching OFF → should have reduced multiplier
  3. If multiplier drops to ~1x, the cache-reuse mechanism is confirmed.

Also measures streaming latency (m=1) in both configs to separate the
prefill cost from the batching headroom.

Usage:
    # Baseline (caching on) — already have results/issue45_baseline.json
    # No-cache streaming — already have results/issue42_nocache.json
    # This script runs the THROUGHPUT comparison:
    python models/experiments/run_issue42_batching.py
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
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record
from metrics.latency_metrics import latency_percentiles

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"


def load_lines(path, n):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def run_batched(llm, sp, prompts, batch_size, repeats=3):
    """Run bursts of batch_size simultaneous requests, measure wall time."""
    cursor = 0
    run_wall = []
    for r in range(repeats):
        if cursor + batch_size > len(prompts):
            cursor = 0
        batch = prompts[cursor:cursor + batch_size]
        cursor += batch_size
        t0 = time.perf_counter()
        outs = llm.generate(batch, sp, use_tqdm=False)
        wall_s = time.perf_counter() - t0
        run_wall.append(wall_s)
    wall_s = statistics.median(run_wall)
    ev_s = batch_size / wall_s
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    tok_s = out_tokens / wall_s
    return {"batch_size": batch_size, "wall_s": round(wall_s, 3),
            "events_per_s": round(ev_s, 1), "tokens_per_s": round(tok_s, 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=256)
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default="results/issue42_batching.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    log_lines = load_lines(str(SPLIT_DIR / "bgl.raw"), args.n)
    print(f"Loaded {len(log_lines)} events", flush=True)

    sp = SamplingParams(max_tokens=args.max_new_tokens,
                        min_tokens=args.min_tokens, temperature=0.0)

    results = {
        "experiment": "issue42_batching_headroom",
        "model": args.model,
        "n_events": len(log_lines),
        "max_new_tokens": args.max_new_tokens,
        "min_tokens": args.min_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "configs": {},
    }

    for config_name, enable_cache in [("caching_on", True), ("caching_off", False)]:
        print(f"\n=== {config_name} ===", flush=True)
        llm = LLM(model=args.model, dtype="bfloat16",
                  gpu_memory_utilization=0.85, max_model_len=4096,
                  enable_prefix_caching=enable_cache)

        tok = llm.get_tokenizer()
        prompts = []
        for line in log_lines:
            msgs = build_chat_messages(line, source="bgl", few_shot=True)
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

        sweep = []
        for b in args.batch_sizes:
            if b > len(prompts):
                continue
            row = run_batched(llm, sp, prompts, b, args.repeats)
            sweep.append(row)
            print(f"  batch {b:4d}: {row['events_per_s']:7.1f} ev/s  "
                  f"{row['tokens_per_s']:8.0f} tok/s", flush=True)

        peak = max(sweep, key=lambda r: r["events_per_s"])
        batch1 = next(r for r in sweep if r["batch_size"] == 1)
        multiplier = peak["events_per_s"] / batch1["events_per_s"]

        results["configs"][config_name] = {
            "sweep": sweep,
            "batch1_ev_s": batch1["events_per_s"],
            "peak_ev_s": peak["events_per_s"],
            "peak_batch": peak["batch_size"],
            "multiplier": round(multiplier, 1),
        }
        print(f"  multiplier: {multiplier:.1f}x "
              f"(batch1={batch1['events_per_s']:.1f}, "
              f"peak={peak['events_per_s']:.1f})", flush=True)

        del llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n=== Comparison ===")
    on = results["configs"]["caching_on"]
    off = results["configs"]["caching_off"]
    print(f"  caching ON:  multiplier={on['multiplier']:.1f}x  "
          f"batch1={on['batch1_ev_s']:.1f} ev/s  "
          f"peak={on['peak_ev_s']:.1f} ev/s")
    print(f"  caching OFF: multiplier={off['multiplier']:.1f}x  "
          f"batch1={off['batch1_ev_s']:.1f} ev/s  "
          f"peak={off['peak_ev_s']:.1f} ev/s")
    print(f"  multiplier drop: {on['multiplier'] / off['multiplier']:.2f}x")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
