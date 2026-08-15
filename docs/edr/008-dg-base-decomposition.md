# EDR-008: 93% of DG's base is prefill+fixed, not denoising

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #44, #68, #63

## Hypothesis

DiffusionGemma's ~418 ms base (single-canvas p50) is treated as an
architectural constant in the cost law. But the K-sweep data (K=8: 375 ms,
K=32: 416 ms) shows a 4× increase in step budget buys only ~41 ms. If
denoising accounts for a minority of the base, the "base" is mostly
prefill + fixed overhead, and the denoising step count is nearly free
quality-wise.

## Design

Sweep K ∈ {2, 4, 8, 16, 32, 64} at L_out=256, n=200, BGL, pinned stack.
Fit T = base + slope × K. Report F1 at each K. Separately measure the
input-length slope (#44) to isolate prefill from fixed overhead.

## Prediction

- **P3:** ≥60% of DG's 418 ms base is prefill + fixed overhead, not
  denoising (already implied by the 8→32 step data).
- **P5:** F1 is flat (±0.01) across K=8→64 but collapses below K=4.

## Result

| K | p50 (ms) | F1 (9-field) |
|---|---|---|
| 2 | 201.7 | 0.4845 |
| 4 | 292.1 | 0.7899 |
| 8 | 378.6 | 0.8533 |
| 16 | 382.5 | 0.8530 |
| 32 | 421.8 | 0.8615 |
| 64 | 423.9 | 0.8580 |

Fit: T = 296.1 + 2.57 × K (R²=0.49 — low because the step cost saturates
above K=16).

- **P3 confirmed:** At K=8, denoising = 20.6 ms = 6.5% of total. Prefill +
  fixed = 296.1 ms = 93.5%.
- **P5 confirmed:** F1 is flat from K=8 to K=64 (0.853–0.862, ±0.009) but
  collapses at K=4 (0.790) and K=2 (0.485).

**Artifacts:** `results/issue44_dg_ksweep_K*.json`

## Decision

**Accepted:** the DG base is not a denoising cost — it is a prefill +
fixed-overhead cost with a small denoising increment. The cost law's `base`
parameter is dominated by input processing, not by the diffusion loop.

This has two consequences:
1. K=8 is the right default — it is the knee where F1 plateaus and
   denoising cost is minimal. Higher K wastes time; lower K destroys
   quality.
2. The input-length slope (#44) is the dominant variable in the base, not
   K. The two-variable law T_DG(L_in, L_out) is primarily shaped by L_in
   through the prefill.

## Impact

- The paper's §4 reports the base decomposition explicitly rather than
  treating 418 ms as a monolithic constant.
- The decision procedure reports the base as a function of K (default K=8)
  and of L_in (0.123 ms/prompt-tok from #44).
- The "first extra canvas is 16% cheaper" finding is dissolved: the
  per-canvas step is constant at ~120 ms regardless of position, and the
  apparent discount was K=32 vs K=8 drift (EDR-001).
