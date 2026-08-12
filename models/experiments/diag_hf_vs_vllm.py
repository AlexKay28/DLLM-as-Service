#!/usr/bin/env python3
"""
Why does AR-1.5B score Field-F1 0.779 under HF transformers and 0.844 under
vLLM, on the same model, prompts, events and greedy decoding?

The gap sits almost entirely in two fields — host (0.202 -> 0.779) and
message_template (0.673 -> 0.854) — and the HF outputs run ~11 tokens shorter,
about the length of a BGL host string. That is a behavioural difference, not
kernel-level numerical drift, so one of the two runs is wrong.

This re-runs the HF path (`ARModel`, the exact class that produced
results/quality_table2_ar_n2000.json) on the first N events and dumps per-event
predictions, so they can be diffed record-by-record against the vLLM
predictions from `run_vllm_serving.py --save-predictions`.

Run under the interpreter the original experiments used (system python3,
transformers 5.14.1 / torch 2.5.1+cu121), not .venv-dllm:

    python3 models/experiments/diag_hf_vs_vllm.py --n 100
    .venv-dllm/bin/python models/experiments/run_vllm_serving.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --n 100 --skip-throughput \
        --save-predictions results/diag_vllm_preds.jsonl \
        --output results/diag_vllm_100.json
    python3 models/experiments/diag_hf_vs_vllm.py --compare
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES
from metrics.quality_metrics import field_f1

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
RESULTS = _REPO_ROOT / "results"
HF_PREDS = RESULTS / "diag_hf_preds.jsonl"
VLLM_PREDS = RESULTS / "diag_vllm_preds.jsonl"


def load_lines(path, n):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                out.append(line.strip())
    return out


def run_hf(n, model_name, max_new_tokens):
    from models.baselines.ar_wrapper import ARModel

    log_lines = load_lines(SPLIT_DIR / "bgl.raw", n)
    model = ARModel(model_name, max_new_tokens=max_new_tokens, batch_size=1)
    model.load()
    result = model.extract(log_lines, source="bgl", show_progress=True)

    with open(HF_PREDS, "w") as f:
        for rec, ms, ntok in zip(result.records, result.latencies_ms,
                                 result.token_counts):
            f.write(json.dumps({
                "record": rec.fields, "latency_ms": ms, "n_tokens": ntok}) + "\n")
    print(f"wrote {HF_PREDS} ({len(result.records)} records, "
          f"avg {sum(result.token_counts)/len(result.token_counts):.1f} tokens)")


def compare():
    gold = [Record(fields=json.loads(l)) for l in
            load_lines(SPLIT_DIR / "bgl.jsonl", 10 ** 9)]

    def read(p):
        return [json.loads(l) for l in load_lines(p, 10 ** 9)]

    hf, vl = read(HF_PREDS), read(VLLM_PREDS)
    n = min(len(hf), len(vl))
    hf, vl, gold = hf[:n], vl[:n], gold[:n]
    print(f"comparing {n} events\n")

    hf_recs = [Record(fields=d["record"]) for d in hf]
    vl_recs = [Record(fields=d["record"]) for d in vl]
    hf_f1, vl_f1 = field_f1(hf_recs, gold), field_f1(vl_recs, gold)

    print(f"{'field':24s} {'HF':>7s} {'vLLM':>7s} {'diff':>7s}")
    for name in FIELD_NAMES:
        k = f"f1_{name}"
        print(f"{name:24s} {hf_f1[k]:7.3f} {vl_f1[k]:7.3f} "
              f"{vl_f1[k]-hf_f1[k]:+7.3f}")
    print(f"{'MACRO':24s} {hf_f1['macro_field_f1']:7.3f} "
          f"{vl_f1['macro_field_f1']:7.3f} "
          f"{vl_f1['macro_field_f1']-hf_f1['macro_field_f1']:+7.3f}\n")

    # Where the two stacks disagree on the field that drives the gap.
    shown = 0
    for i, (h, v, g) in enumerate(zip(hf, vl, gold)):
        hh, vh = h["record"].get("host", ""), v["record"].get("host", "")
        if hh == vh:
            continue
        shown += 1
        if shown <= 5:
            print(f"[{i}] gold host = {g.get('host')!r}")
            print(f"      HF   = {hh!r}")
            print(f"      vLLM = {vh!r}")
    print(f"\nhost differs on {shown}/{n} events")

    hf_empty = sum(1 for d in hf if not d["record"].get("host"))
    vl_empty = sum(1 for d in vl if not d["record"].get("host"))
    print(f"empty host: HF {hf_empty}/{n}, vLLM {vl_empty}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--compare", action="store_true",
                    help="skip inference, just diff the two prediction files")
    args = ap.parse_args()

    if args.compare:
        compare()
    else:
        run_hf(args.n, args.model, args.max_new_tokens)


if __name__ == "__main__":
    main()
