# Issue #12: AR baselines under production serving (vLLM)

**Date:** 2026-08-12 &nbsp;|&nbsp; **GPU:** H100 PCIe 80GB &nbsp;|&nbsp; **Driver:** 580.65.06 (CUDA 13.0) &nbsp;|&nbsp; **vLLM:** 0.19.1, torch 2.10.0+cu128

**Data:** BGL test split, n=2000 (same events as the HF-transformers runs, so quality is directly comparable).

## Why this run exists

Every AR number previously in this repo came from a naive HuggingFace `model.generate()` loop: one event at a time, no PagedAttention, no continuous batching, no prefix reuse. The paper serves its AR baselines with PagedAttention + FlashAttention-2 and explicitly credits them with a frozen-Task conditioning cache — precisely so diffusion does not win on serving infrastructure. Our AR side was therefore a strawman, and every AR-vs-MDM latency conclusion drawn from it was confounded.

## Table 3: Streaming latency (m=1)

| Serving | Model | p50 (ms) | p99 (ms) | p99.9 (ms) | p99.9/p50 | Field-F1 |
|---|---|---|---|---|---|---|
| **vLLM** | **AR-1.5B** | **413.3** | **685.5** | **805.8** | **1.95x** | **0.844** |
| **vLLM** | **AR-3B** | **662.0** | **1062.1** | **1336.3** | **2.02x** | **0.826** |
| **vLLM** | **AR-7B** | **1161.1** | **1521.1** | **2310.9** | **1.99x** | **0.842** |
| HF transformers | AR-1.5B | 1841.6 | 3053.4 | 4162.5 | 2.26x | 0.779 |
| HF transformers | AR-7B | 2122.0 | 5479.7 | 7808.0 | 3.68x | 0.839 |
| HF transformers | MDM LLaDA-8B (K=8) | 1389.0 | 1555.5 | 1648.2 | 1.19x | 0.779 |
| *paper* | *AR-1.5B* | 168.0 | 240.0 | 340.0 | 2.02x | 0.913 |
| *paper* | *AR-7B* | 690.0 | 910.0 | 1240.0 | 1.80x | 0.945 |
| *paper* | *FlowFill-MDM (K=8)* | 23.0 | 31.0 | 36.0 | 1.57x | 0.907 |

## Table 4: Throughput under continuous batching

Each row is a burst of B requests submitted at once; p99 is the burst wall-clock, i.e. when the last request finished. Bursts draw fresh log lines so prefix caching serves only the shared few-shot preamble, not whole prompts.

### AR-1.5B

| Batch | p99 (ms) | events/s | tokens/s |
|---|---|---|---|
| 1 | 418.6 | 2.4 | 332 |
| 2 | 448.7 | 4.5 | 649 |
| 4 | 469.6 | 8.5 | 1212 |
| 8 | 510.5 | 15.7 | 2247 |
| 16 | 538.3 | 29.7 | 4053 |
| 32 | 634.1 | 50.5 | 6871 |
| 64 | 970.4 | 66.0 | 9279 |
| 128 | 1141.1 | 112.2 | 15348 |
| 256 | 1561.8 | 163.9 | 22626 |
| 512 | 2498.5 | 204.9 | 28250 |

Max batch meeting each p99 SLA:

| SLA (ms) | max batch | events/s |
|---|---|---|
| 250 | none | — |
| 500 | 4 | 8.5 |
| 1000 | 64 | 66.0 |
| 2000 | 256 | 163.9 |

Peak: **204.9 ev/s** at batch 512 (28250 tok/s).

### AR-3B

| Batch | p99 (ms) | events/s | tokens/s |
|---|---|---|---|
| 1 | 695.3 | 1.4 | 193 |
| 2 | 682.7 | 2.9 | 379 |
| 4 | 719.6 | 5.6 | 723 |
| 8 | 756.5 | 10.6 | 1339 |
| 16 | 903.3 | 17.7 | 2385 |
| 32 | 1186.6 | 27.0 | 3550 |
| 64 | 1378.3 | 46.4 | 6153 |
| 128 | 1788.9 | 71.6 | 9460 |
| 256 | 2189.4 | 116.9 | 15549 |
| 512 | 3527.2 | 145.2 | 19277 |

Max batch meeting each p99 SLA:

| SLA (ms) | max batch | events/s |
|---|---|---|
| 250 | none | — |
| 500 | none | — |
| 1000 | 16 | 17.7 |
| 2000 | 128 | 71.6 |

Peak: **145.2 ev/s** at batch 512 (19277 tok/s).

### AR-7B

| Batch | p99 (ms) | events/s | tokens/s |
|---|---|---|---|
| 1 | 1199.3 | 0.8 | 111 |
| 2 | 1179.4 | 1.7 | 220 |
| 4 | 1219.9 | 3.3 | 427 |
| 8 | 1296.6 | 6.2 | 795 |
| 16 | 1516.6 | 10.6 | 1392 |
| 32 | 1713.4 | 18.7 | 2447 |
| 64 | 2278.2 | 28.1 | 3747 |
| 128 | 2360.5 | 54.2 | 7173 |
| 256 | 3560.8 | 71.9 | 9448 |
| 512 | 6015.0 | 85.1 | 11267 |

Max batch meeting each p99 SLA:

| SLA (ms) | max batch | events/s |
|---|---|---|
| 250 | none | — |
| 500 | none | — |
| 1000 | none | — |
| 2000 | 32 | 18.7 |

Peak: **85.1 ev/s** at batch 512 (11267 tok/s).

Paper's Table 4 reference: FlowFill 4350 ev/s @ 250ms p99 SLA.

## Frozen-T conditioning cache (prefix caching)

Our prompt is a fixed ~1k-token system + few-shot preamble followed by the one log line that varies. That is exactly the paper's frozen-T block, and prefix caching is how vLLM realises it.

| Prefix caching | p50 (ms) | p99 (ms) | p99.9 (ms) |
|---|---|---|---|
| on (paper-consistent) | 413.3 | 685.5 | 805.8 |
| off | 425.6 | 818.4 | 823.8 |

Cost of dropping the cache: 1.03x on p50.

## The AR baseline was also handicapped by a decoding default

Chasing a Field-F1 gap between the two stacks (0.779 HF vs 0.844 vLLM on the same model, prompts, events and greedy decoding) turned up a bug affecting every HF-transformers AR result in this repo.

Qwen2.5 ships a `repetition_penalty` in `generation_config.json` — 1.1 for 1.5B, 1.05 for 3B and 7B — and HF `generate()` applies it even under `do_sample=False`. `ARModel` never set the parameter, so the model's own default silently took effect. For this task the penalty is actively wrong: correct extraction *copies spans out of the input*, which is precisely what a repetition penalty suppresses. vLLM's `SamplingParams` defaults it to 1.0, which is why only the HF side was affected.

Same code path, same 100 BGL events, only the penalty changed:

| | HF, penalty 1.1 (model default) | HF, penalty 1.0 | vLLM |
|---|---|---|---|
| host F1 | 0.200 | **0.800** | 0.830 |
| message_template F1 | 0.691 | **0.854** | 0.873 |
| macro Field-F1 | 0.778 | **0.849** | 0.853 |
| empty `host` | 80/100 | 20/100 | 17/100 |
| mean output tokens | 128.9 | 137.9 | 138.2 |

With the penalty neutralised the two stacks agree to 0.004 — ordinary bf16 kernel drift. Only the copy-from-input fields moved; severity, component and the anomaly slots are untouched, which is the signature the penalty predicts.

Fixed in `ar_wrapper.py`: `repetition_penalty` is now an explicit constructor argument defaulting to 1.0, so the decoding config is an experimental choice rather than whatever a given checkpoint ships.

**This understated every AR number in the repo**, worst at 1.5B where the penalty is strongest — the paper's size-matched comparison. The MDM side was never affected: LLaDA has no `repetition_penalty` in its config and the denoising loop is hand-written. So the AR baseline was handicapped relative to MDM on quality, just as it was on serving.

## What this changes

1. **Serving, not architecture, dominated our AR numbers.** AR-1.5B p50 goes 1841.6ms → 413.3ms (4.5x) purely by swapping HF `generate()` for vLLM. Same model, same weights, same prompts, same events.

2. **The reported MDM latency win does not survive.** The earlier report concluded MDM was 1.33x faster than AR (LLaDA-8B 1389ms vs AR-1.5B 1842ms). With AR served properly, AR-1.5B is 3.4x *faster* than LLaDA-8B. That comparison was measuring HF `generate()` against a hand-written denoising loop, not AR against MDM.

3. **The bounded-tail claim is untouched by this.** It rests on the tail *ratio*, and MDM's O(K) content-independent latency still gives it the tightest tail in the table. Proposition 1 survives; the absolute speed claim it was bundled with does not.

4. **"AR-vs-MDM near-parity" does not survive either.** That finding paired AR-1.5B at 0.779 with LLaDA-8B at 0.779 — but the AR half of that pair was the penalty-suppressed number. Corrected, AR reaches 0.844 (AR-1.5B) against MDM's 0.779, a gap of 0.065. The parity was an artefact of handicapping AR, not a property of the two architectures.

## Caveats

1. **This does not show AR beats MDM.** It shows the two sides were served unequally. LLaDA-8B is still on naive HF transformers — there is no MDM row here served under vLLM, because vLLM's only MDM (DiffusionGemma, from 0.24.0) pins torch 2.11/CUDA 13 and this host's mirror has no CUDA-13 cuDNN/NCCL/cuFile/cuSPARSELt wheels that `libtorch_cuda.so` hard-links. See `results/diffusiongemma_perf.md`.
2. **Different hardware from the paper** (H100 PCIe vs A100-80GB PCIe): ratios are comparable, absolute numbers are not.
3. **Output length differs from the paper's budget.** We generate up to 256 tokens and average ~128; the paper's AR baseline is 128 steps. At 256-token outputs no AR configuration clears a 250ms p99, which is why the SLA table reports a grid rather than the single 250ms point.

---
*Generated by `models/experiments/make_vllm_report.py`*
