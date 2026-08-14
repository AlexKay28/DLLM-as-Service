"""
Issue #22 smoke test: load LLaDA-8B-Instruct under dInfer and generate on a
handful of BGL events. Verifies the framework loads + produces parseable
12-field records before running the full correctness/latency/batch gates.

Single-process (mirrors benchmarks/benchmark.py dense `llada` branch), NOT the
multi-process DiffusionLLMServing API — easier to debug, and dllm.generate()
accepts a batch directly so P-F2 (batch sweep) can use the same path.

Usage:
    .venv-dinfer/bin/python models/experiments/smoke_dinfer.py [--n 10] [--gen-len 256]
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

from metrics.schema import Record, FIELD_NAMES, FIELD_BY_NAME, FieldType
from metrics.quality_metrics import field_f1
from models.baselines.ar_model import build_chat_messages, parse_json_output
from models.baselines.mdm_wrapper import _dict_to_record

MODEL_PATH = ("/home/alexkay28/.cache/huggingface/hub/models--GSAI-ML--"
              "LLaDA-8B-Instruct/snapshots/"
              "08b83a6feb34df1a6011b80c3c00c7563e963b07/")
SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"

# LLaDA-8B-Instruct special token ids (confirmed via tokenizer +
# dInfer benchmark.py dense `llada` branch).
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
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--cache", default="", help="'' | 'prefix' | 'dual'")
    parser.add_argument("--cont-weight", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true", help="torch.compile model.forward")
    parser.add_argument("--source", default="bgl")
    parser.add_argument("--few-shot", dest="few_shot", action="store_true", default=True)
    parser.add_argument("--no-few-shot", dest="few_shot", action="store_false")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoConfig
    from dinfer.model import LLaDAModelLM
    from dinfer.decoding.serving import SamplingParams, init_generator

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print(f"Loading {args.n} {args.source} events...")
    log_lines = load_log_lines(SPLIT_DIR / f"{args.source}.raw", args.n)
    gold = load_records(SPLIT_DIR / f"{args.source}.jsonl", args.n)
    print(f"  {len(log_lines)} lines, {len(gold)} gold records")

    print(f"Loading tokenizer from {MODEL_PATH} ...")
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
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params; "
          f"GPU mem {torch.cuda.memory_allocated()/1e9:.1f}GB")

    sp = SamplingParams(
        mask_id=MASK_ID, eos_id=EOS_ID,
        cache=args.cache, cont_weight=args.cont_weight,
        threshold=args.threshold, temperature=0.0,
        parallel_decoding="threshold", early_stop=True,
        use_bd=False, enable_torch_compile=False,
    )
    dllm = init_generator(model, sp, backend="vllm", max_length=4096)

    # warmup (cudagraph / lazy init)
    print("Warmup generate ...")
    wu = tok("warmup prompt", return_tensors="pt")["input_ids"].to(device)
    t0 = time.time()
    dllm.generate(wu, gen_length=args.block_length, block_length=args.block_length)
    print(f"  warmup done in {time.time()-t0:.1f}s")

    records, lats, out_toks = [], [], []
    for i, line in enumerate(log_lines):
        msgs = build_chat_messages(line, source=args.source, few_shot=args.few_shot)
        prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        input_ids = tok(prompt_text, return_tensors="pt",
                       truncation=True, max_length=2048)["input_ids"].to(device)
        plen = input_ids.shape[1]
        if i == 0:
            print(f"  prompt_len={plen} (few_shot={args.few_shot})")

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
        print(f"[{i:2d}] {elapsed:7.1f}ms  out_tok={n_out:3d}  "
              f"lat={lats[-1]:.0f}  text[:100]={text[:100]!r}")

    f1 = field_f1(records, gold)
    lats_sorted = sorted(lats)
    print("\n=== smoke summary ===")
    print(f"n={len(records)}  Field-F1={f1['macro_field_f1']:.4f}")
    print(f"lat ms: p50={lats_sorted[len(lats_sorted)//2]:.0f}  "
          f"p99={lats_sorted[-1]:.0f}  "
          f"mean={sum(lats)/len(lats):.0f}")
    print(f"out tokens: mean={sum(out_toks)/len(out_toks):.0f}")
    # show one parsed record vs gold
    print("\n=== sample record ===")
    print("pred:", json.dumps({k: records[0].get(k) for k in FIELD_NAMES},
                             default=str)[:300])
    print("gold:", json.dumps({k: gold[0].get(k) for k in FIELD_NAMES},
                             default=str)[:300])


if __name__ == "__main__":
    main()
