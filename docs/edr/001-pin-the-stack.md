# EDR-001: Pin the measurement stack to vLLM 0.25.1

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #57, #63

## Hypothesis

Cross-run comparison in this repo is currently invalid because five distinct
software stacks have produced numbers that sit in the same tables. We
hypothesise that pinning to a single stack will (a) explain the two anchor
discrepancies and (b) produce run-to-run spread under 2%.

## Design

Pin to vLLM 0.25.1 (`.venv-dg`, torch 2.11.0+cu130, A100 80GB PCIe). This is
the only stack that supports DiffusionGemma (via `diffusion_config` in
`EngineArgs`, added in vLLM 0.24.0). Re-measure four anchor cells (AR-3B
L=256/512, DG L=256/512) with 5 repeats each, n=300/cell.

A shared provenance helper (`models/baselines/provenance.py`) records the
stack inside every result JSON. No runner can omit it.

## Prediction

- The AR-3B 1617.6 → 1915.4 ms drift is a stack difference (vLLM 0.25.1 vs
  0.7.2), not noise.
- The DG 418.24 → 378.8 ms drift is a configuration difference (K=32 vs K=8),
  not stack drift.
- Run-to-run CV on all four anchor cells will be < 2%.

## Result

| Cell | Repeats | p50 median (ms) | CV (%) |
|---|---|---|---|
| AR-3B L=256 | 5 | 1616.0 | 0.05 |
| AR-3B L=512 | 3 | 3225.1 | 0.01 |
| DG L=256 | 5 | 379.0 | 0.25 |
| DG L=512 | 5 | 500.7 | 0.12 |

All predictions confirmed:
- AR drift explained: vLLM 0.25.1 reproduces 1617 ms; 0.7.2 is 18% slower.
- DG drift explained: p2_dg cells match K=32 (416 ms); dg_steps_8 is K=8 (375 ms).
- All CVs far below 2%.

**Artifacts:** `results/anchors/anchor_remeasure.json`,
`results/anchors/ar3b_anchor_summary.json`

## Decision

**Accepted:** vLLM 0.25.1 is the pinned stack. All new measurements use it.
The 0.7.2 stack is retained for provenance but not cited in tables.

## Impact

- Every subsequent experiment inherits the pinned stack and the provenance
  helper.
- The 18% AR gap is documented in `docs/legacy/paper/NUMBERS.md` §0 — it is a stack
  property, not a model property.
- K=8 is pinned for all DG cost-law measurements. K=32 is retained for the
  ablation (Table 5) only.
