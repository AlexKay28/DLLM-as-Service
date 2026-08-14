"""
Issue #22 — P-F2: dInfer batch multiplier sweep for LLaDA-8B.

The number #17 needs: does dInfer's batch multiplier scale (AR-like ~67x) or
stay weak (~2-3x, Fast-dLLM's measured 102.5->217.5 tok/s across batch 1->4)?

Method: for each batch size B, replicate ONE prompt B times -> a (B, L) batch
of identical-length sequences (no padding, no attention-mask complications).
Time K timed generates; ev/s = B / median_batch_time. The multiplier is
ev_s[B] / ev_s[1]. Prompt content is irrelevant to the multiplier (it is a
ratio of throughputs for the same per-event workload).

Uses cache='' (the prefix/vicinity cache errors on our prompt lengths; see
run_dinfer_serving.py docstring). Uses --no-few-shot (632-token prompt) to
keep the sweep affordable; the multiplier is length-independent.

Usage:
    .venv-dinfer/bin/python models/experiments/run_dinfer_batch.py \
        --batches 1 2 4 8 16 32 64 --reps 3
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.baselines.ar_model import build_chat_messages

MODEL_PATH = ("/home/alexkay28/.cache/huggingface/hub/models--GSAI-ML--"
              "LLaDA-8B-Instruct/snapshots/"
              "08b83a6feb34df1a6011b80c3c00c7563e963b07/")
SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
RESULTS_DIR = _REPO_ROOT / "results" / "llada_dinfer"

MASK_ID = 126336
EOS_ID = 126081


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--reps", type=int, default=3,
                        help="timed generates per batch size (take median)")
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--cache", default="")
    parser.add_argument("--source", default="bgl")
    parser.add_argument("--out", default="batch_sweep.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoConfig
    from dinfer.model import LLaDAModelLM
    from dinfer.decoding.serving import SamplingParams, init_generator

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # one representative prompt (few_shot=False -> ~632 tokens)
    log_lines = load_log_lines(SPLIT_DIR / f"{args.source}.raw", 1)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        local_files_only=True)
    msgs = build_chat_messages(log_lines[0], source=args.source, few_shot=False)
    prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt",
                     truncation=True, max_length=2048)["input_ids"]
    L = prompt_ids.shape[1]
    print(f"prompt_len={L} (few_shot=False)")

    print("Loading LLaDAModelLM (dInfer) ...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, config=config).eval()
    model.init_h2e_module()
    model = model.to(device)
    torch.cuda.synchronize()
    print(f"  loaded in {time.time()-t0:.1f}s ; "
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params ; "
          f"GPU {torch.cuda.memory_allocated()/1e9:.1f}GB")

    sp = SamplingParams(
        mask_id=MASK_ID, eos_id=EOS_ID,
        cache=args.cache, cont_weight=0.0,
        threshold=args.threshold, temperature=0.0,
        parallel_decoding="threshold", early_stop=True,
        use_bd=False, enable_torch_compile=False,
    )
    dllm = init_generator(model, sp, backend="vllm", max_length=4096)

    # warmup at batch 1
    print("Warmup ...")
    wu = prompt_ids.to(device)
    dllm.generate(wu, gen_length=args.block_length, block_length=args.block_length)
    torch.cuda.synchronize()
    # warmup at the largest batch to cudagraph/compile any batch-specific state
    big = prompt_ids.expand(max(args.batches), -1).contiguous().to(device)
    dllm.generate(big, gen_length=args.gen_len, block_length=args.block_length)
    torch.cuda.synchronize()
    print("  warmup done")

    results = []
    base_ev_s = None
    for B in args.batches:
        batch = prompt_ids.expand(B, -1).contiguous().to(device)
        times = []
        n_outs = []
        for r in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = dllm.generate(batch, gen_length=args.gen_len,
                                block_length=args.block_length)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            n_outs.append(int((out[0, L:] != EOS_ID).sum()))
        med = statistics.median(times)
        ev_s = B / med
        tok_s = ev_s * (sum(n_outs) / len(n_outs))
        if base_ev_s is None:
            base_ev_s = ev_s
        mult = ev_s / base_ev_s if base_ev_s else 1.0
        results.append({
            "batch": B,
            "median_time_s": round(med, 4),
            "ev_s": round(ev_s, 3),
            "tok_s": round(tok_s, 1),
            "multiplier": round(mult, 3),
            "times_s": [round(t, 4) for t in times],
            "avg_out_tokens": round(sum(n_outs) / len(n_outs), 1),
        })
        print(f"  B={B:3d}  {med:.3f}s  {ev_s:7.2f} ev/s  {tok_s:8.1f} tok/s  "
              f"mult={mult:.2f}x")

    peak = max(results, key=lambda r: r["ev_s"])
    out_obj = {
        "model": "LLaDA-8B-Instruct",
        "serving": "dInfer 0.1 (cache='', BlockWiseDiffusionLLM, threshold=0.9)",
        "host": "A100 80GB PCIe, vLLM 0.10.2 stack, torch 2.8.0+cu128",
        "prompt_len": L,
        "few_shot": False,
        "gen_len": args.gen_len,
        "block_length": args.block_length,
        "cache": args.cache,
        "results": results,
        "batch1_ev_s": base_ev_s,
        "peak_ev_s": peak["ev_s"],
        "peak_batch": peak["batch"],
        "batch_multiplier": round(peak["ev_s"] / base_ev_s, 3) if base_ev_s else None,
    }
    out_path = RESULTS_DIR / args.out
    with open(out_path, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\nBatch multiplier (peak/B=1): {out_obj['batch_multiplier']}x "
          f"(peak {peak['ev_s']} ev/s @ B={peak['batch']}, "
          f"B=1 {base_ev_s:.2f} ev/s)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
