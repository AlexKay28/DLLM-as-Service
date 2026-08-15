# EDR-014: Canvas is a tunable config parameter

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #66, #63

## Hypothesis

The DG canvas size (256 tokens) is a configuration choice, not a model
property. If it is exposed as a runtime knob, shrinking it should lower the
base floor and move the length crossover left, widening the DG regime.

## Design

Step 0 (10 minutes): read DiffusionGemma's `config.json` and vLLM's
`diffusion_config` parameter. Step 1 (GPU): sweep canvas ∈ {256, 128, 64},
measure p50, 9-field F1, schema validity. Kill condition: if base drops
<15% when shrinking 256→64, stop and report negative.

## Prediction

- Canvas is tunable (config field exists).
- base(128) < base(256), base(64) < base(128).
- Crossover moves left proportionally.

## Result

`config.json` contains `canvas_length: 256`. vLLM accepts it via
`diffusion_config={"canvas_length": N}`. The knob exists.

| Canvas | p50 (ms) | F1 | vs 256 |
|---|---|---|---|
| 256 | 376.0 | 0.8533 | baseline |
| 128 | 368.5 | 0.8644 | base -2.4%, F1 +0.011 |
| 64 | 557.5 | 0.8266 | base +48%, F1 -0.027 |

Kill condition fired for canvas=64. Canvas=128 is a free improvement but
does not meaningfully move the crossover (L* shifts from 59 to 57).

LLaDA-8B uses `block_group_size` (not `canvas_length`) — different mechanism,
same concept.

**Artifacts:** `results/issue66_canvas_{256,128,64}.json`,
DiffusionGemma `config.json`

## Decision

**Accepted:** canvas is tunable but not a design lever. The crossover is set
by prefill cost (93% of base, EDR-008), which is canvas-independent. Canvas=128
is recommended as a deployment default (strictly better, no trade-off).

## Impact

- C16 goes green (canvas is a config parameter, confirmed and measured).
- The paper does not claim the crossover is tunable. It reports canvas=128
  as a minor deployment recommendation.
- The paper's genre remains "verdict" not "design recommendation" — the
  positive result that would have changed the genre did not materialise.
