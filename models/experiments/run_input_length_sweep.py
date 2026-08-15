#!/usr/bin/env python3
"""Issue #44 — input-length sweep: are cost laws T(L_out) or T(L_in, L_out)?

Sweeps prompt length by prepending variable-length filler context (repeated
schema descriptions) to the standard prompt. Measures latency at each input
length with fixed output length (min=max=256 tokens).

If T(L_in, L_out) = a + b*L_in + c*L_out, then varying L_in at fixed L_out
should reveal the b coefficient. If b~0, the cost law is single-variable.

Supports both AR (Qwen2.5) and DG (DiffusionGemma) models.

Usage:
    # AR-3B
    python models/experiments/run_input_length_sweep.py --n 100

    # DiffusionGemma (K=8)
    python models/experiments/run_input_length_sweep.py \
        --model google/diffusiongemma-26B-A4B-it --denoising-steps 8 --n 100
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
from models.baselines.ar_model import (
    build_chat_messages, parse_json_output, SYSTEM_PROMPT, SCHEMA_DESCRIPTION,
)
from models.baselines.ar_wrapper import _dict_to_record
from models.baselines.provenance import capture_provenance

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


def load_gold(path, n):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                records.append(Record(fields=json.loads(line)))
    return records


def make_padded_prompt(tokenizer, log_line, source, n_schema_repeats):
    """Build a prompt with n_schema_repeats copies of the schema description
    prepended to the user content, to sweep input length."""
    msgs = build_chat_messages(log_line, source=source, few_shot=True)
    if n_schema_repeats > 0:
        padding = "\n\nContext:\n" + SCHEMA_DESCRIPTION + "\n"
        msgs[1]["content"] = padding * n_schema_repeats + msgs[1]["content"]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=256)
    ap.add_argument("--schema-repeats", type=int, nargs="+",
                    default=[0, 2, 5, 10, 20, 40],
                    help="number of extra schema-description copies to prepend")
    ap.add_argument("--denoising-steps", type=int, default=None,
                    help="DG only: max_denoising_steps (e.g. 8)")
    ap.add_argument("--output", default="results/issue44_input_sweep.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    log_lines = load_lines(str(SPLIT_DIR / "bgl.raw"), args.n)
    gold = load_gold(str(SPLIT_DIR / "bgl.jsonl"), args.n)
    print(f"Loaded {len(log_lines)} events", flush=True)

    llm_kwargs = dict(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=8192,
        enable_prefix_caching=False,
    )
    if args.denoising_steps is not None:
        llm_kwargs["diffusion_config"] = {
            "canvas_length": 256,
            "max_denoising_steps": args.denoising_steps,
        }
    llm = LLM(**llm_kwargs)

    model_type = "dg" if "diffusion" in args.model.lower() else "ar"
    prov = capture_provenance(
        model_name=args.model, model_type=model_type, backend="vllm",
        max_new_tokens=args.max_new_tokens,
        parser_policy="first_object_wins", scoring_fields="9",
    )
    if args.denoising_steps is not None:
        prov.denoising_steps = args.denoising_steps

    sp = SamplingParams(max_tokens=args.max_new_tokens,
                        min_tokens=args.min_tokens, temperature=0.0)

    results = {
        "experiment": "issue44_input_length_sweep",
        "model": args.model,
        "model_type": model_type,
        "n_events": len(log_lines),
        "max_new_tokens": args.max_new_tokens,
        "min_tokens": args.min_tokens,
        "denoising_steps": args.denoising_steps,
        "provenance": prov.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sweep": [],
    }

    tok = llm.get_tokenizer()

    for n_rep in args.schema_repeats:
        prompts = [make_padded_prompt(tok, line, "bgl", n_rep)
                   for line in log_lines]
        prompt_tok = len(tok.encode(prompts[0]))

        latencies = []
        records = []
        out_tokens = []
        print(f"\n  schema_repeats={n_rep}, prompt_tok={prompt_tok} ...",
              flush=True)
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            outs = llm.generate([prompt], sp, use_tqdm=False)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            text = outs[0].outputs[0].text
            out_tokens.append(len(outs[0].outputs[0].token_ids))
            records.append(_dict_to_record(parse_json_output(text)))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(prompts)}  "
                      f"p50={statistics.median(latencies):.1f}ms", flush=True)

        f1_res = field_f1(records, gold[:len(records)])
        entry = {
            "schema_repeats": n_rep,
            "prompt_token_count": prompt_tok,
            "latency_ms": {
                "p50": round(statistics.median(latencies), 1),
                "p99": round(sorted(latencies)[int(len(latencies)*0.99)], 1),
                "mean": round(statistics.fmean(latencies), 1),
            },
            "mean_output_tokens": round(statistics.fmean(out_tokens), 1),
            "field_f1": round(f1_res["macro_field_f1_9"], 4),
            "field_f1_9": round(f1_res["macro_field_f1_9"], 4),
            "field_f1_12": round(f1_res["macro_field_f1_12"], 4),
            "schema_valid_pct": round(schema_validity(records), 2),
        }
        results["sweep"].append(entry)
        print(f"  p50={entry['latency_ms']['p50']:.1f}ms  "
              f"prompt_tok={prompt_tok}  F1={entry['field_f1']:.3f}",
              flush=True)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"{'repeats':>7s} {'prompt_tok':>11s} {'p50_ms':>8s} {'p99_ms':>8s} "
          f"{'ms/tok_out':>11s} {'F1':>5s}")
    for e in results["sweep"]:
        ms_per_tok = e["latency_ms"]["p50"] / e["mean_output_tokens"]
        print(f"{e['schema_repeats']:7d} {e['prompt_token_count']:11d} "
              f"{e['latency_ms']['p50']:8.1f} {e['latency_ms']['p99']:8.1f} "
              f"{ms_per_tok:11.2f} {e['field_f1']:5.3f}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
