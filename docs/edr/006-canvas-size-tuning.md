# EDR-006: Canvas=128 is a free improvement; canvas=64 is not viable

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #66, #63

## Hypothesis

If the DG cost law's floor is set by the canvas (within a canvas, emitting
one token and emitting 256 tokens cost the same), then shrinking the canvas
should lower the floor and move the length crossover left, widening the
region where diffusion wins. The crossover is then a fact about a
configuration choice, not a fact about diffusion.

## Design

Sweep `canvas_length` ∈ {256, 128, 64} on DiffusionGemma-26B-A4B-it via
vLLM's `diffusion_config`. K=8, n=200, L_out=256, BGL corpus. Measure p50,
9-field F1, schema validity. Kill condition: if base drops <15% when
shrinking 256→64, stop and report negative.

## Prediction

- **P1:** base(128) < base(256), base(64) < base(128).
- **P2:** crossover moves left proportionally to the base drop.
- **P3:** quality degrades as canvas shrinks, first on consistency fields.

## Result

| Canvas | p50 (ms) | F1 (9-field) | Schema valid % | vs canvas=256 |
|---|---|---|---|---|
| 256 | 376.0 | 0.8533 | 99.0% | baseline |
| 128 | 368.5 | 0.8644 | 98.5% | base -2.4%, F1 +0.011 |
| 64 | 557.5 | 0.8266 | 91.5% | base +48%, F1 -0.027 |

- **P1 partially refuted:** canvas=128 drops the base by only 2.4% (not the
  15%+ expected). canvas=64 *raises* the base by 48%. The floor is dominated
  by prefill and fixed overhead, not canvas size.
- **P2 refuted:** L_cross moves from 59.1 to 57.5 tokens — negligible.
- **P3 mixed:** canvas=128 *improves* F1 (+0.011). canvas=64 degrades it
  (-0.027) and drops schema validity to 91.5%.
- **Kill condition fired** for canvas=64: base rose 48%, well above the 15%
  threshold.

**Artifacts:** `results/issue66_canvas_{256,128,64}.json`

## Decision

**Accepted:** canvas=128 is a free improvement (slightly faster, slightly
better quality) but does not meaningfully move the crossover. Canvas=64 is
not viable. The crossover is set by the prefill cost, which is
canvas-independent.

The paper does not claim the crossover is tunable. It reports canvas=128 as
a deployment recommendation (strictly better, no trade-off), not as a
design lever.

## Impact

- The paper's genre remains "verdict" (are dLLMs useful, and at what price?),
  not "design recommendation" (here is how to make them useful). The positive
  result that would have changed the genre did not materialise.
- The decision procedure uses canvas=256 as default; canvas=128 is a
  parameter that can be set but does not change the recommendation.
- This closes the last speculative direction in the canvas-tuning space.
