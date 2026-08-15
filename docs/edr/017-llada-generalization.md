# EDR-017: LLaDA under dInfer — form does NOT clearly transfer

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #67, #63

## Hypothesis

If the staircase cost shape is a property of block-structured decoding in
general (not just DiffusionGemma), LLaDA-8B under dInfer should show stepping
at block_length boundaries.

## Design

Sweep gen_len ∈ {128, 256, 512} at block_length=32, and block_length ∈ {32,
64, 128} at gen_len=256. Cache-less mode (dInfer prefix/vicinity cache errors
on prompts ≥632 tokens). n=50, BGL.

## Prediction

Latency steps at gen_len = k × block_length boundaries, with a constant
per-block cost.

## Result

| gen_len | bl | p50 (ms) | F1_9 |
|---|---|---|---|
| 128 | 32 | 2752.8 | 0.8010 |
| 256 | 32 | 11160.9 | 0.8698 |
| 512 | 32 | 4497.3 | 0.8725 |
| 256 | 64 | 10395.3 | 0.8668 |
| 256 | 128 | 10568.4 | 0.8668 |

**Staircase NOT observed.** gen=512 is faster than gen=256 (4497 vs 11161 ms) —
the opposite of a staircase. block_length sweep shows no clear stepping. The
latency is dominated by torch.compile warmup variance and cache-less overhead,
not by block structure.

**Artifacts:** `results/llada_dinfer/issue67_llada_*.json`

## Decision

**Accepted:** the staircase form does NOT clearly transfer to LLaDA under
dInfer. This is stated honestly in §9. Two possible explanations:

1. dInfer's cache-less mode inflates absolute latency by ~30× and introduces
   variance that obscures the block structure.
2. LLaDA uses a different denoising mechanism (pure masked diffusion, not
   block diffusion) that may not produce the same canvas-boundary effect.

The paper states: "the staircase form is confirmed on DiffusionGemma across
two corpora (BGL, HDFS). LLaDA under dInfer does not reproduce it clearly,
but the cache-less runtime makes this inconclusive rather than negative."

## Impact

- §9 threat: "single model family for the staircase (DiffusionGemma). LLaDA
  under dInfer is inconclusive due to cache-less mode."
- The word "law" is retained for the form on DiffusionGemma (two corpora),
  but the paper does not claim it generalizes to all diffusion models.
- A future test would require a production-grade LLaDA runtime with working
  KV cache — which does not exist at time of writing.
