# EDR-010: The fair comparison — where DM wins and where AR wins

**Status:** Proposed
**Date:** 2026-08-15
**Issues:** #63, #41

## The comparison pair

To claim "DM is better/worse than AR," the comparison must be fair on two axes:

1. **Same quality**: 9-field F1 within 0.01
2. **Same compute power**: comparable active inference parameters

The pair that satisfies both:

| | AR-3B | DG-26B-A4B |
|---|---|---|
| Architecture | Dense, autoregressive | MoE, block diffusion |
| Total params | 3.0B | 26B |
| **Active params** | **3.0B** (all active) | **4.0B** (top-8 of 128 experts) |
| Active param ratio | 1.0× | 1.33× |
| Weights (bf16) | 6.0 GB | 52.0 GB |
| 9-field F1 (L=256) | 0.8516 | 0.8518 |

Quality is identical (gap = 0.0002). Active inference params are comparable
(1.33× ratio). Per-forward-pass compute is in the same order of magnitude.

This is NOT the quality-matched deployment pair (AR-1.5B, which is 17× smaller).
This is the **compute-matched inference pair** — the fair comparison for
"which architecture is faster at the same quality and same compute."

## DM is better than AR on:

### a) Per-event latency at batch-1, short input

| L (tokens) | AR-3B (ms) | DG (ms) | DG speedup |
|---|---|---|---|
| 128 | 812 | 378 | 2.15× |
| 256 | 1616 | 378 | 4.28× |
| 512 | 3225 | 501 | 6.44× |
| 1024 | 6448 | 739 | 8.73× |
| 2048 | 12889 | 1212 | 10.63× |

**Why:** AR pays per-token (T = 7.27 + 6.29·L). DG pays per-canvas (T = 379 +
119.6·(⌈L/256⌋−1)). Within a canvas, DG emits 1 token or 256 tokens for the
same cost. AR's cost grows linearly with every token. The longer the output,
the more DG's flat-within-canvas cost amortizes.

### b) Output-length invariance within a canvas

DG at L=32, L=64, L=128, L=256 all cost ~378 ms (spread: 2.5 ms).
AR at the same lengths costs 210 → 416 → 823 → 1616 ms (linear growth).

**Why:** the canvas boundary is a structural property of block-structured
decoding. The model denoises the entire canvas in K=8 parallel steps
regardless of how many tokens are actually needed. AR generates one token at a
time, paying for each.

### c) Latency growth rate with output length

DG's per-canvas step cost is constant at ~120 ms (verified on BGL and HDFS,
#67). AR's per-token cost is constant at ~6.3 ms/token. The ratio:

- At L=256 (1 canvas): DG step = 0, AR decode = 1611 ms → 4.3× advantage
- At L=2048 (8 canvases): DG steps = 7×120 = 840 ms, AR decode = 12882 ms →
  10.6× advantage

**Why:** each additional canvas adds ~120 ms to DG. Each additional 256 tokens
adds ~1613 ms to AR. The gap widens with output length because AR's marginal
cost (6.3 ms/token × 256 = 1613 ms) is 13× higher than DG's marginal cost
(120 ms per canvas).

## AR is better than DM on:

### d) Batched throughput (λ > 1.8 ev/s)

| Config | Batch multiplier | At λ=64 ev/s |
|---|---|---|
| AR-3B (vLLM) | 77–86× | effective 20 ms/ev |
| DG (dInfer) | 1.18× | effective 321 ms/ev |

AR under batching is 16× faster than DG at high arrival rate.

**Why:** AR's KV cache allows batching to amortize the prefill across requests.
DG's bidirectional attention invalidates KV-cache reuse — each denoising step
re-attends the full canvas independently. AR converts spare GPU bandwidth into
throughput; DG is already compute-bound at B=1 and has no bandwidth headroom to
convert. (Note: this is the measured correlation; the causal mechanism is not
isolated — EDR-004.)

### e) Long input (L_in > ~18,000 tokens)

| Input length | AR-3B (ms) | DG (ms) | Winner |
|---|---|---|---|
| 1,000 | 1705 | 420 | DG (4.1×) |
| 4,000 | 1846 | 789 | DG (2.3×) |
| 7,000 | 1987 | 1158 | DG (1.7×) |
| 18,000 | 2504 | 2511 | crossover |
| 25,000 | 2833 | 3372 | AR (1.2×) |

**Why:** AR pays for the prompt once (prefill). DG pays for the prompt on
every denoising step — K=8 steps × full attention over the prompt each time.
DG's input-length slope is 0.123 ms/prompt-tok vs AR's 0.047 (2.6× steeper,
EDR-005). At long input, DG's prefill cost overwhelms its canvas advantage.

### f) Deployment footprint at the quality-matched pair

| | AR-1.5B | DG-26B |
|---|---|---|
| 9-field F1 | 0.8628 | 0.8518 |
| Weights | 3.0 GB | 52.0 GB |
| Cold start | ~1s | ~17s |
| Energy/event | 113 J | (higher, pending measurement) |

AR-1.5B is quality-matched (actually higher F1), 17× smaller, and faster to
start. No deployment axis favours DG.

**Why:** the compute-matched pair (AR-3B vs DG-26B-A4B, 3B vs 4B active) is
NOT the quality-matched pair (AR-1.5B vs DG, 1.5B vs 26B). A smaller AR model
achieves the same quality with 17× less memory. The per-event speed
advantage (a, b, c) does not survive the deployment comparison because a
quality-equivalent AR model is available at a fraction of the cost.

## That's why

The cost shapes produce three regimes:

1. **DG wins**: batch-1, short input, long output. The canvas makes output
   length nearly free within a block, and AR's per-token cost accumulates
   linearly. This is a real per-event latency advantage (2–10×).

2. **AR wins**: batched serving, long input, or any deployment-budget-
   constrained scenario. AR's KV cache gives 80× batch multiplier; AR pays
   for input once; and the quality-matched AR model is 17× smaller.

3. **The crossover**: at L≈60 tokens (output) and λ≈2.6 ev/s (arrival rate).
   Below both: DG. Above either: AR. The region where DG wins is real but
   narrow, and it does not survive the deployment comparison.

The paper's contribution is the decision procedure that draws these
boundaries — not a claim that either architecture is universally better. The
boundaries are the finding.
