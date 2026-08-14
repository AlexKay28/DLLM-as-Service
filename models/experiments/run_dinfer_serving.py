"""
Issue #22 — LLaDA-8B under dInfer: correctness gate (P-F3) + streaming
latency (P-F1), single process.

P-F3: Field-F1 must reproduce the HF baseline (0.779) within noise.
P-F1: p50/p99/p99.9 vs LLaDA-HF (1389/1556/1648 ms) and AR-3B-vLLM.

Note on the cache: dInfer's prefix/vicinity KV-cache paths
(VicinityCacheDiffusionLLM / BlockWiseDiffusionLLM+cache) error on our prompt
lengths (>=632 tokens) — the LLaDA attention `slice_scatter` mismatches the
windowed-forward key width against the cache slice. This is a dInfer
limitation for long prompts (dInfer was benchmarked on ~100-token prompts).
We therefore run cache='' (BlockWiseDiffusionLLM, no cache). This is a FAIR
comparison to the HF baseline, which also ran cache-less denoising
(mdm_wrapper.py:273, `use_cache=False`).

Usage:
    .venv-dinfer/bin/python models/experiments/run_dinfer_serving.py \
        --n 300 --cache '' --gen-len 256 --block-length 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES
from metrics.quality_metrics import field_f1, schema_validity, consistency, anomaly_f1
from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.mdm_wrapper import _dict_to_record

MODEL_PATH = ("/home/alexkay28/.cache/huggingface/hub/models--GSAI-ML--"
              "LLaDA-8B-Instruct/snapshots/"
              "08b83a6feb34df1a6011b80c3c00c7563e963b07/")
SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
RESULTS_DIR = _REPO_ROOT / "results" / "llada_dinfer"

MASK_ID = 126336
EOS_ID = 126081

# HF baseline (VALIDATION_REPORT.md §5.1) for P-F1 comparison.
HF_BASELINE = {"p50_ms": 1389, "p99_ms": 1556, "p999_ms": 1648, "field_f1": 0.779}


def load_log_lines(path, n):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                out.append(line)
    return out


def load_records(path, n):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                out.append(Record(fields=json.loads(line)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--cache", default="", help="'' | 'prefix' | 'dual'")
    parser.add_argument("--cont-weight", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--source", default="bgl")
    parser.add_argument("--few-shot", dest="few_shot", action="store_true", default=True)
    parser.add_argument("--no-few-shot", dest="few_shot", action="store_false")
    parser.add_argument("--out", default="correctness_latency.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoConfig
    from dinfer.model import LLaDAModelLM
    from dinfer.decoding.serving import SamplingParams, init_generator

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n} {args.source} events...")
    log_lines = load_log_lines(SPLIT_DIR / f"{args.source}.raw", args.n)
    gold = load_records(SPLIT_DIR / f"{args.source}.jsonl", args.n)
    print(f"  {len(log_lines)} lines, {len(gold)} gold records")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        local_files_only=True)
    print("Loading LLaDAModelLM (dInfer) ...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, config=config).eval()
    model.init_h2e_module()
    model = model.to(device)
    if args.compile:
        model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                      fullgraph=False, dynamic=True)
    torch.cuda.synchronize()
    print(f"  loaded in {time.time()-t0:.1f}s ; "
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params ; "
          f"GPU {torch.cuda.memory_allocated()/1e9:.1f}GB")

    sp = SamplingParams(
        mask_id=MASK_ID, eos_id=EOS_ID,
        cache=args.cache, cont_weight=args.cont_weight,
        threshold=args.threshold, temperature=0.0,
        parallel_decoding="threshold", early_stop=True,
        use_bd=False, enable_torch_compile=False,
    )
    dllm = init_generator(model, sp, backend="vllm", max_length=4096)

    # warmup
    print("Warmup ...")
    wu = tok("warmup prompt", return_tensors="pt")["input_ids"].to(device)
    t0 = time.time()
    dllm.generate(wu, gen_length=args.block_length, block_length=args.block_length)
    torch.cuda.synchronize()
    print(f"  warmup done in {time.time()-t0:.1f}s")

    records, lats, out_toks, in_toks = [], [], [], []
    t_start = time.time()
    for i, line in enumerate(log_lines):
        msgs = build_chat_messages(line, source=args.source, few_shot=args.few_shot)
        prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        input_ids = tok(prompt_text, return_tensors="pt",
                       truncation=True, max_length=2048)["input_ids"].to(device)
        plen = input_ids.shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = dllm.generate(input_ids, gen_length=args.gen_len,
                           block_length=args.block_length)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        gen = out[0][plen:]
        n_out = int((gen != EOS_ID).sum())
        text = tok.decode(gen, skip_special_tokens=True)
        data = parse_json_output(text)
        rec = _dict_to_record(data)
        records.append(rec)
        lats.append(elapsed)
        out_toks.append(n_out)
        in_toks.append(plen)
        if (i + 1) % 25 == 0 or i == 0:
            avg = sum(lats) / len(lats)
            print(f"  [{i+1}/{len(log_lines)}] avg={avg:.0f}ms "
                  f"last={elapsed:.0f}ms out_tok={n_out}")

    total = time.time() - t_start

    # --- quality (P-F3) ---
    f1_res = field_f1(records, gold)
    sv = schema_validity(records)
    cons = consistency(records)
    anom = anomaly_f1(records, gold)
    # --- latency (P-F1) ---
    lat = latency_percentiles(lats)

    result = {
        "model": "LLaDA-8B-Instruct",
        "serving": "dInfer 0.1 (cache='', BlockWiseDiffusionLLM, threshold=0.9)",
        "host": "A100 80GB PCIe, vLLM 0.10.2 stack, torch 2.8.0+cu128",
        "source": args.source,
        "n_events": len(records),
        "few_shot": args.few_shot,
        "gen_len": args.gen_len,
        "block_length": args.block_length,
        "cache": args.cache,
        "field_f1": round(f1_res["macro_field_f1"], 4),
        "per_field_f1": {k: round(v, 4) for k, v in f1_res.items()
                         if k.startswith("f1_")},
        "schema_valid_pct": round(sv, 2),
        "consistency_pct": round(cons, 2),
        "anomaly_f1": round(anom["anomaly_f1"], 4),
        "p50_ms": round(lat.get("p50_ms", 0), 2),
        "p99_ms": round(lat.get("p99_ms", 0), 2),
        "p999_ms": round(lat.get("p99_9_ms", 0), 2),
        "mean_ms": round(sum(lats) / len(lats), 2),
        "min_ms": round(min(lats), 2),
        "max_ms": round(max(lats), 2),
        "total_time_s": round(total, 1),
        "avg_in_tokens": round(sum(in_toks) / len(in_toks), 1),
        "avg_out_tokens": round(sum(out_toks) / len(out_toks), 1),
        "throughput_ev_s": round(len(records) / total, 3),
        "hf_baseline": HF_BASELINE,
    }
    out_path = RESULTS_DIR / args.out
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== P-F3 correctness gate ===")
    print(f"  Field-F1:       {result['field_f1']:.4f}  (HF baseline 0.779)")
    print(f"  Schema-valid:   {result['schema_valid_pct']:.1f}%")
    print(f"  Consistency:    {result['consistency_pct']:.1f}%")
    print(f"  Anomaly-F1:     {result['anomaly_f1']:.4f}")
    print("=== P-F1 latency ===")
    print(f"  p50/p99/p99.9:  {result['p50_ms']:.0f}/{result['p99_ms']:.0f}/"
          f"{result['p999_ms']:.0f} ms  (HF 1389/1556/1648)")
    print(f"  mean={result['mean_ms']:.0f}ms  throughput={result['throughput_ev_s']:.2f} ev/s")
    print(f"  avg tokens in/out: {result['avg_in_tokens']:.0f}/{result['avg_out_tokens']:.0f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
