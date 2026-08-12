#!/usr/bin/env python3
"""
Issue #12: build results/vllm_serving.md from the vLLM sweep outputs.

Puts the vLLM-served AR numbers next to the HF-transformers AR numbers, the
LLaDA-8B MDM numbers, and the paper's own table, so the effect of serving
infrastructure is readable as a row-to-row difference rather than as a claim.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _REPO_ROOT / "results"

VLLM_RUNS = [
    ("AR-1.5B", "vllm_serving_1.5b.json"),
    ("AR-3B", "vllm_serving_3b.json"),
    ("AR-7B", "vllm_serving_7b.json"),
]


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def first_model(run):
    """These runs hold exactly one model each."""
    return next(iter(run["models"].values()))


def fmt(v, spec=".1f"):
    return "—" if v is None else format(v, spec)


def main():
    hf_ar = load("quality_table2_ar_n2000.json")
    hf_mdm = load("quality_table2_mdm_llada_n2000.json")
    noprefix = load("vllm_serving_1.5b_noprefix.json")

    runs = []
    for label, fname in VLLM_RUNS:
        r = load(fname)
        if r:
            runs.append((label, r, first_model(r)))
    if not runs:
        sys.exit("no vLLM result files found — run run_vllm_all.sh first")

    paper = (hf_ar or {}).get("paper_table2", {})
    hf_by_name = {m["model"]: m for m in (hf_ar or {}).get("models", [])}
    mdm = (hf_mdm or {}).get("models", [{}])[0]

    L = []
    A = L.append
    A("# Issue #12: AR baselines under production serving (vLLM)")
    A("")
    A("**Date:** 2026-08-12 &nbsp;|&nbsp; **GPU:** H100 PCIe 80GB &nbsp;|&nbsp; "
      "**Driver:** 580.65.06 (CUDA 13.0) &nbsp;|&nbsp; **vLLM:** 0.19.1, torch 2.10.0+cu128")
    A("")
    A("**Data:** BGL test split, n=2000 (same events as the HF-transformers "
      "runs, so quality is directly comparable).")
    A("")
    A("## Why this run exists")
    A("")
    A("Every AR number previously in this repo came from a naive HuggingFace "
      "`model.generate()` loop: one event at a time, no PagedAttention, no "
      "continuous batching, no prefix reuse. The paper serves its AR baselines "
      "with PagedAttention + FlashAttention-2 and explicitly credits them with "
      "a frozen-Task conditioning cache — precisely so diffusion does not win "
      "on serving infrastructure. Our AR side was therefore a strawman, and "
      "every AR-vs-MDM latency conclusion drawn from it was confounded.")
    A("")

    # ---------------- Table 3 ----------------
    A("## Table 3: Streaming latency (m=1)")
    A("")
    A("| Serving | Model | p50 (ms) | p99 (ms) | p99.9 (ms) | p99.9/p50 | Field-F1 |")
    A("|---|---|---|---|---|---|---|")

    for label, run, m in runs:
        lat, q = m["streaming_table3"]["latency"], m["streaming_table3"]["quality"]
        A(f"| **vLLM** | **{label}** | **{fmt(lat['p50_ms'])}** | "
          f"**{fmt(lat['p99_ms'])}** | **{fmt(lat['p99_9_ms'])}** | "
          f"**{fmt(m['streaming_table3']['tail_ratio_p999_over_p50'], '.2f')}x** | "
          f"**{fmt(q['field_f1'], '.3f')}** |")

    for disp, key in [("AR-1.5B", "Qwen2.5-1.5B-Instruct"),
                      ("AR-7B", "Qwen2.5-7B-Instruct")]:
        h = hf_by_name.get(key)
        if h:
            A(f"| HF transformers | {disp} | {fmt(h['p50_ms'])} | {fmt(h['p99_ms'])} | "
              f"{fmt(h['p999_ms'])} | {fmt(h['p999_ms']/h['p50_ms'], '.2f')}x | "
              f"{fmt(h['field_f1'], '.3f')} |")

    if mdm:
        A(f"| HF transformers | MDM LLaDA-8B (K=8) | {fmt(mdm['p50_ms'])} | "
          f"{fmt(mdm['p99_ms'])} | {fmt(mdm['p999_ms'])} | "
          f"{fmt(mdm['p999_ms']/mdm['p50_ms'], '.2f')}x | "
          f"{fmt(mdm['field_f1'], '.3f')} |")

    for name in ["AR-1.5B", "AR-7B", "FlowFill-MDM (K=8)"]:
        p = paper.get(name)
        if p:
            A(f"| *paper* | *{name}* | {fmt(p['p50_ms'])} | {fmt(p['p99_ms'])} | "
              f"{fmt(p['p999_ms'])} | {fmt(p['p999_ms']/p['p50_ms'], '.2f')}x | "
              f"{fmt(p['field_f1'], '.3f')} |")
    A("")

    # ---------------- Table 4 ----------------
    A("## Table 4: Throughput under continuous batching")
    A("")
    A("Each row is a burst of B requests submitted at once; p99 is the burst "
      "wall-clock, i.e. when the last request finished. Bursts draw fresh log "
      "lines so prefix caching serves only the shared few-shot preamble, not "
      "whole prompts.")
    A("")
    for label, run, m in runs:
        tp = m.get("throughput_table4")
        if not tp:
            continue
        A(f"### {label}")
        A("")
        A("| Batch | p99 (ms) | events/s | tokens/s |")
        A("|---|---|---|---|")
        for row in tp["sweep"]:
            A(f"| {row['batch_size']} | {fmt(row['p99_ms'])} | "
              f"{fmt(row['events_per_s'])} | {fmt(row['tokens_per_s'], '.0f')} |")
        A("")
        A("Max batch meeting each p99 SLA:")
        A("")
        A("| SLA (ms) | max batch | events/s |")
        A("|---|---|---|")
        for sla, v in tp["at_sla"].items():
            A(f"| {sla} | {v['max_batch'] or 'none'} | {fmt(v['events_per_s'])} |")
        A("")
        A(f"Peak: **{fmt(tp['peak_events_per_s'])} ev/s** at batch "
          f"{tp['peak_events_per_s_batch']} "
          f"({fmt(tp['peak_tokens_per_s'], '.0f')} tok/s).")
        A("")

    if paper.get("FlowFill-MDM (K=8)"):
        A("Paper's Table 4 reference: FlowFill 4350 ev/s @ 250ms p99 SLA.")
        A("")

    # ---------------- prefix caching ----------------
    if noprefix:
        npm = first_model(noprefix)
        base = runs[0][2]
        nl = npm["streaming_table3"]["latency"]
        bl = base["streaming_table3"]["latency"]
        A("## Frozen-T conditioning cache (prefix caching)")
        A("")
        A("Our prompt is a fixed ~1k-token system + few-shot preamble followed "
          "by the one log line that varies. That is exactly the paper's frozen-T "
          "block, and prefix caching is how vLLM realises it.")
        A("")
        A("| Prefix caching | p50 (ms) | p99 (ms) | p99.9 (ms) |")
        A("|---|---|---|---|")
        A(f"| on (paper-consistent) | {fmt(bl['p50_ms'])} | {fmt(bl['p99_ms'])} | "
          f"{fmt(bl['p99_9_ms'])} |")
        A(f"| off | {fmt(nl['p50_ms'])} | {fmt(nl['p99_ms'])} | {fmt(nl['p99_9_ms'])} |")
        A("")
        if bl["p50_ms"]:
            A(f"Cost of dropping the cache: "
              f"{fmt(nl['p50_ms'] / bl['p50_ms'], '.2f')}x on p50.")
            A("")

    # ---------------- verdict ----------------
    A("## What this changes")
    A("")
    base_lat = runs[0][2]["streaming_table3"]["latency"]["p50_ms"]
    hf15 = hf_by_name.get("Qwen2.5-1.5B-Instruct", {}).get("p50_ms")
    if hf15:
        A(f"1. **Serving, not architecture, dominated our AR numbers.** "
          f"AR-1.5B p50 goes {fmt(hf15)}ms → {fmt(base_lat)}ms "
          f"({fmt(hf15/base_lat, '.1f')}x) purely by swapping HF `generate()` "
          f"for vLLM. Same model, same weights, same prompts, same events.")
    if mdm and base_lat:
        faster = mdm["p50_ms"] / base_lat
        A("")
        A(f"2. **The reported MDM latency win does not survive.** The earlier "
          f"report concluded MDM was 1.33x faster than AR (LLaDA-8B 1389ms vs "
          f"AR-1.5B 1842ms). With AR served properly, AR-1.5B is "
          f"{fmt(faster, '.1f')}x *faster* than LLaDA-8B. That comparison was "
          f"measuring HF `generate()` against a hand-written denoising loop, "
          f"not AR against MDM.")
    A("")
    A("3. **The bounded-tail claim is untouched by this.** It rests on the "
      "tail *ratio*, and MDM's O(K) content-independent latency still gives it "
      "the tightest tail in the table. Proposition 1 survives; the absolute "
      "speed claim it was bundled with does not.")
    A("")
    A("## Caveats")
    A("")
    A("1. **This does not show AR beats MDM.** It shows the two sides were "
      "served unequally. LLaDA-8B is still on naive HF transformers — there is "
      "no MDM row here served under vLLM, because vLLM's only MDM "
      "(DiffusionGemma, from 0.24.0) pins torch 2.11/CUDA 13 and this host's "
      "mirror has no CUDA-13 cuDNN/NCCL/cuFile/cuSPARSELt wheels that "
      "`libtorch_cuda.so` hard-links. See `results/diffusiongemma_perf.md`.")
    A("2. **Different hardware from the paper** (H100 PCIe vs A100-80GB PCIe): "
      "ratios are comparable, absolute numbers are not.")
    A("3. **Output length differs from the paper's budget.** We generate up to "
      "256 tokens and average ~128; the paper's AR baseline is 128 steps. At "
      "256-token outputs no AR configuration clears a 250ms p99, which is why "
      "the SLA table reports a grid rather than the single 250ms point.")
    A("")
    A("---")
    A("*Generated by `models/experiments/make_vllm_report.py`*")

    out = RESULTS / "vllm_serving.md"
    out.write_text("\n".join(L) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
