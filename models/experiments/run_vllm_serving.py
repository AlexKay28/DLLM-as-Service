#!/usr/bin/env python3
"""
Issue #12: AR baselines under production-grade serving (vLLM).

Every previous AR number in this repo came from a naive HuggingFace
`model.generate()` loop — one event at a time, no PagedAttention, no continuous
batching, no prefix reuse across events. That makes the AR side a strawman: the
paper's own AR baselines are served with PagedAttention + FlashAttention-2 and
are explicitly credited with a frozen-Task conditioning cache, precisely so the
diffusion-vs-AR comparison is not won on serving infrastructure.

This runner re-measures the AR side with vLLM so that:

  Table 3 (latency)     — p50/p99/p99.9 in streaming mode (m=1, one request in
                          flight) are attributable to the model, not to the
                          serving loop.
  Table 4 (throughput)  — events/s under a fixed p99 SLA is measured with real
                          continuous batching instead of a sequential loop
                          pretending to be a batch.

Prefix caching maps onto the paper's frozen-T conditioning cache: our prompt is
a fixed ~1k-token system + few-shot preamble followed by the one log line that
varies, so with caching on, only the log line is prefilled per event. It is on
by default (paper-consistent) and `--no-prefix-caching` measures the cost of
turning it off.

DiffusionGemma note: this runner deliberately covers only the AR side. vLLM
gained DiffusionGemma (its only MDM) in 0.24.0, which pins torch 2.11 / CUDA 13,
and this host's mirror carries no CUDA-13 cuDNN/NCCL/cuFile/cuSPARSELt wheels
that libtorch_cuda.so hard-links. See results/diffusiongemma_perf.md.

Usage:
    python models/experiments/run_vllm_serving.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --n 2000

    # size-matched sweep
    python models/experiments/run_vllm_serving.py \
        --model Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct --n 2000
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
from metrics.quality_metrics import (
    field_f1, schema_validity, consistency, anomaly_f1,
)
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.ar_wrapper import _dict_to_record

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


def summarize_latency(values):
    """Repo-standard percentiles plus the spread, so rows here line up with
    every earlier Table 3 result rather than shifting under a new estimator."""
    out = dict(latency_percentiles(values))
    if values:
        out["mean_ms"] = float(statistics.fmean(values))
        out["min_ms"] = float(min(values))
        out["max_ms"] = float(max(values))
    out["n_samples"] = len(values)
    return out


def build_prompts(llm, log_lines, source):
    """Render chat-templated prompts, identical to what ARModel feeds HF."""
    tokenizer = llm.get_tokenizer()
    prompts = []
    for line in log_lines:
        messages = build_chat_messages(line, source=source, few_shot=True)
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
    return prompts


def run_streaming(llm, sampling_params, prompts, gold_records, log_every=100,
                  save_predictions=None, save_raw=False):
    """Table 3: one request in flight at a time (streaming mode, m=1)."""
    latencies_ms = []
    records = []
    out_tokens = []
    raw_texts = []

    print(f"  streaming {len(prompts)} events (m=1)...", flush=True)
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sampling_params, use_tqdm=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        text = outs[0].outputs[0].text
        out_tokens.append(len(outs[0].outputs[0].token_ids))
        records.append(_dict_to_record(parse_json_output(text)))
        if save_predictions is not None:
            raw_texts.append(text)

        if (i + 1) % log_every == 0:
            print(f"    {i+1}/{len(prompts)}  "
                  f"p50={statistics.median(latencies_ms):.1f}ms", flush=True)

    if save_predictions is not None:
        with open(save_predictions, "w") as f:
            for rec, txt, ms in zip(records, raw_texts, latencies_ms):
                f.write(json.dumps({
                    "record": rec.fields, "raw": txt, "latency_ms": ms}) + "\n")
        print(f"    wrote predictions to {save_predictions}", flush=True)

    lat = summarize_latency(latencies_ms)
    gold = gold_records[:len(records)]
    f1_res = field_f1(records, gold)
    quality = {
        "field_f1": round(f1_res["macro_field_f1"], 4),
        "per_field_f1": {k: round(v, 4) for k, v in f1_res.items()
                         if k.startswith("f1_")},
        "schema_valid_pct": round(schema_validity(records), 2),
        "consistency_pct": round(consistency(records), 2),
        "anomaly_f1": round(anomaly_f1(records, gold)["anomaly_f1"], 4),
    }
    result = {
        "latency": lat,
        "quality": quality,
        "mean_output_tokens": float(statistics.fmean(out_tokens)),
        "tail_ratio_p999_over_p50": (
            lat["p99_9_ms"] / lat["p50_ms"] if lat.get("p50_ms") else None),
    }
    if save_raw:
        result["raw_latencies_ms"] = latencies_ms
        result["raw_out_tokens"] = out_tokens
        per_event_f1 = []
        for rec, g in zip(records, gold):
            r = field_f1([rec], [g])
            per_event_f1.append(r["macro_field_f1"])
        result["per_event_f1"] = per_event_f1
    return result


def run_throughput(llm, sampling_params, prompts, batch_sizes, sla_grid, repeats):
    """Table 4: burst of B simultaneous requests through continuous batching.

    Per-request latency is taken from the engine's own arrival/finish
    timestamps when it reports them. When it does not, every request in the
    burst arrives at t=0, so the burst wall-clock is exactly the latency of the
    last request to finish — i.e. an upper bound on p99 — and that is what gets
    recorded. `latency_source` in the output says which one was used, because
    the two are not interchangeable when reading a p99 off this table.

    Each burst draws a fresh window of log lines via a rolling cursor. Reusing
    the same slice would let prefix caching serve whole prompts, not just the
    shared few-shot preamble, and the throughput would then partly measure the
    cache rather than the model.
    """
    sweep = []
    cursor = 0
    for b in batch_sizes:
        if b > len(prompts):
            print(f"    batch {b}: skipped (only {len(prompts)} prompts loaded)",
                  flush=True)
            continue

        run_wall = []
        run_p99 = []
        run_out_tokens = []
        latency_source = "burst_wall_clock"
        for r in range(repeats):
            if cursor + b > len(prompts):
                cursor = 0
            batch = prompts[cursor:cursor + b]
            cursor += b
            t0 = time.perf_counter()
            outs = llm.generate(batch, sampling_params, use_tqdm=False)
            wall_s = time.perf_counter() - t0
            run_wall.append(wall_s)
            run_out_tokens.append(
                sum(len(o.outputs[0].token_ids) for o in outs))

            per_req = []
            for o in outs:
                m = getattr(o, "metrics", None)
                if m and getattr(m, "finished_time", None) and getattr(m, "arrival_time", None):
                    per_req.append((m.finished_time - m.arrival_time) * 1000.0)
            if len(per_req) == len(outs) and per_req:
                latency_source = "engine_per_request"
                run_p99.append(latency_percentiles(per_req)["p99_ms"])
            else:
                run_p99.append(wall_s * 1000.0)

        wall_s = statistics.median(run_wall)
        p99_ms = statistics.median(run_p99)
        ev_s = b / wall_s
        row = {
            "batch_size": b,
            "wall_s": wall_s,
            "p99_ms": p99_ms,
            "events_per_s": ev_s,
            "tokens_per_s": statistics.median(run_out_tokens) / wall_s,
            "latency_source": latency_source,
        }
        sweep.append(row)
        print(f"    batch {b:4d}: wall={wall_s:6.2f}s  p99={p99_ms:8.1f}ms  "
              f"{ev_s:7.1f} ev/s  {row['tokens_per_s']:8.0f} tok/s", flush=True)

    # The paper's 250ms SLA is reported alongside looser ones: at 256 output
    # tokens no AR config can clear 250ms, so a single-threshold table would
    # read "0 ev/s" and hide the actual shape of the latency/throughput curve.
    at_sla = {}
    for sla in sla_grid:
        meeting = [r for r in sweep if r["p99_ms"] <= sla]
        best = max(meeting, key=lambda r: r["events_per_s"]) if meeting else None
        at_sla[str(int(sla))] = {
            "max_batch": best["batch_size"] if best else None,
            "events_per_s": best["events_per_s"] if best else None,
            "tokens_per_s": best["tokens_per_s"] if best else None,
        }

    peak = max(sweep, key=lambda r: r["events_per_s"]) if sweep else None
    return {
        "sweep": sweep,
        "at_sla": at_sla,
        "peak_events_per_s": peak["events_per_s"] if peak else None,
        "peak_events_per_s_batch": peak["batch_size"] if peak else None,
        "peak_tokens_per_s": peak["tokens_per_s"] if peak else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--source", default="bgl",
                    choices=["hdfs", "bgl", "thunderbird", "spirit"])
    ap.add_argument("--n", type=int, default=500,
                    help="events for the streaming latency/quality pass")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="force every request to generate at least this many "
                         "tokens (decode-mechanics scaling experiments; "
                         "changes task semantics — do not use for quality "
                         "measurements)")
    ap.add_argument("--sla-ms", type=float, nargs="+",
                    default=[250.0, 500.0, 1000.0, 2000.0],
                    help="p99 SLA thresholds; 250ms is the paper's Table 4 SLA")
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--throughput-repeats", type=int, default=3)
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="disable prefix caching (drops the paper's frozen-T "
                         "conditioning cache credit)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--denoising-steps", type=int, default=None,
                    help="dLLM only: max_denoising_steps for the diffusion "
                         "canvas (DiffusionGemma default is 48, from its "
                         "generation_config.json). Sweep K to trace the "
                         "latency-in-K curve.")
    ap.add_argument("--output", default="results/vllm_serving.json")
    ap.add_argument("--save-predictions",
                    help="JSONL path for per-event predictions (one model only)")
    ap.add_argument("--save-raw", action="store_true",
                    help="save per-request latency array + per-event F1 in the "
                         "output JSON for bootstrap CIs (issue #46)")
    ap.add_argument("--speculative-config", default=None,
                    help="JSON dict for vLLM speculative decoding config, e.g. "
                         "'{\"method\":\"ngram\",\"num_speculative_tokens\":5,"
                         "\"prompt_lookup_max\":4}' (issue #45)")
    ap.add_argument("--guided-json", default=None,
                    help="path to a JSON Schema file for grammar-constrained "
                         "decoding via vLLM GuidedDecodingParams (issue #47)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    log_lines = load_log_lines(str(SPLIT_DIR / f"{args.source}.raw"), n=args.n)
    gold_records = load_records(str(SPLIT_DIR / f"{args.source}.jsonl"), n=args.n)
    print(f"Loaded {len(log_lines)} {args.source} events "
          f"({len(gold_records)} gold records)")

    results = {
        "experiment": "issue12_vllm_serving",
        "source": args.source,
        "n_events": len(log_lines),
        "sla_grid_ms": args.sla_ms,
        "max_new_tokens": args.max_new_tokens,
        "min_tokens": args.min_tokens,
        "prefix_caching": not args.no_prefix_caching,
        "serving": "vLLM (PagedAttention + continuous batching)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {},
    }

    for model_name in args.model:
        print(f"\n=== {model_name} ===", flush=True)
        llm_kwargs = dict(
            model=model_name,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enable_prefix_caching=not args.no_prefix_caching,
        )
        if args.denoising_steps is not None:
            # A user-provided DiffusionConfig is used as-is (no merge with the
            # checkpoint's config.json), so canvas_length must be restated or
            # it defaults to None and crashes the runner's buffer alloc.
            llm_kwargs["diffusion_config"] = {
                "canvas_length": 256,
                "max_denoising_steps": args.denoising_steps}
        llm = LLM(**llm_kwargs)
        sp_kwargs = dict(
            max_tokens=args.max_new_tokens, min_tokens=args.min_tokens,
            temperature=0.0)
        if args.speculative_config:
            import json as _json
            spec = _json.loads(args.speculative_config)
            sp_kwargs["speculative_config"] = spec
            print(f"  speculative decoding: {spec}", flush=True)
        if args.guided_json:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                with open(args.guided_json) as gf:
                    schema = _json.load(gf)
                sp_kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
                print(f"  grammar-constrained decoding: {args.guided_json}",
                      flush=True)
            except ImportError:
                print("  WARNING: GuidedDecodingParams not available in this "
                      "vLLM version", flush=True)
        sampling_params = SamplingParams(**sp_kwargs)

        prompts = build_prompts(llm, log_lines, args.source)
        entry = {"model": model_name}
        if args.denoising_steps is not None:
            entry["max_denoising_steps"] = args.denoising_steps
        try:
            import vllm as _vllm
            entry["vllm_version"] = getattr(_vllm, "__version__", "unknown")
        except Exception:
            entry["vllm_version"] = "unknown"
        try:
            import torch as _torch
            entry["torch_version"] = getattr(_torch, "__version__", "unknown")
        except Exception:
            entry["torch_version"] = "unknown"

        prompts = build_prompts(llm, log_lines, args.source)
        try:
            tok = llm.get_tokenizer()
            entry["prompt_token_count"] = len(tok.encode(prompts[0])) if prompts else 0
        except Exception:
            entry["prompt_token_count"] = None

        entry["streaming_table3"] = run_streaming(
            llm, sampling_params, prompts, gold_records,
            save_predictions=args.save_predictions, save_raw=args.save_raw)
        lat = entry["streaming_table3"]["latency"]
        q = entry["streaming_table3"]["quality"]
        print(f"  p50={lat['p50_ms']:.1f}ms  p99={lat['p99_ms']:.1f}ms  "
              f"p99.9={lat['p99_9_ms']:.1f}ms  "
              f"tail={entry['streaming_table3']['tail_ratio_p999_over_p50']:.2f}x")
        print(f"  Field-F1={q['field_f1']:.3f}  "
              f"schema-valid={q['schema_valid_pct']:.1f}%  "
              f"consistency={q['consistency_pct']:.1f}%")

        if not args.skip_throughput:
            print("  throughput sweep...", flush=True)
            entry["throughput_table4"] = run_throughput(
                llm, sampling_params, prompts, args.batch_sizes,
                args.sla_ms, args.throughput_repeats)
            tp = entry["throughput_table4"]
            print(f"  peak {tp['peak_events_per_s']:.1f} ev/s at batch "
                  f"{tp['peak_events_per_s_batch']}", flush=True)

        results["models"][model_name] = entry

        # Free the GPU before the next model loads.
        del llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
