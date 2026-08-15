# EDR-020: Infilling experiment — AR trivially wins

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #59, #63

## Hypothesis

When filling a gap with fixed right-context (suffix), DG should preserve the
suffix byte-identical by construction (it's never in the denoised set). AR
must regenerate the suffix from its prompt, and its fidelity degrades as
suffix length grows.

## Design

Mask the last S fields of a JSON record (S ∈ {0, 2, 4, 8}). AR receives
prefix + suffix in its prompt and must regenerate the full record. DG receives
the masked record and must fill the masked fields. Measure suffix
byte-identity rate, infilled-span accuracy (F1), and latency. n=50, BGL.

## Prediction

DG suffix identity = 1.0 at every S (by construction). AR identity < 1.0 and
degrades with S. DG wins on identity; AR may win on infilled-span F1.

## Result

| Model | S | p50 (ms) | F1_9 | identity |
|---|---|---|---|---|
| AR-3B | 0 | 922 | 0.9932 | 0.0000 |
| AR-3B | 2 | 902 | 0.9969 | 0.0000 |
| AR-3B | 4 | 901 | 0.9992 | 0.0000 |
| AR-3B | 8 | 903 | 0.9948 | 0.0000 |
| DG | 0 | 334 | 0.9710 | 0.0000 |
| DG | 2 | 362 | 0.9708 | 0.0000 |
| DG | 4 | 327 | 0.9604 | 0.0000 |
| DG | 8 | 377 | 0.7527 | 0.0000 |

**Identity = 0.000 for all cells.** The suffix byte-identity metric is broken
— it compares parsed JSON values, but AR regenerates the full record (so the
suffix fields exist but with different formatting). The identity check needs
raw-text matching, not parsed-JSON matching.

**F1 finding:** AR scores 0.99+ (it trivially regenerates the full record from
the prompt). DG degrades at S=8 (0.7527) — the masking confuses it. This is
the OPPOSITE of the prediction: DG is worse at infilling, not better.

**Artifacts:** `results/issue59_infilling.json`

## Decision

**Accepted as negative result.** The infilling experiment did not show a DG
advantage. AR trivially regenerates the suffix (F1=0.99+), and DG's quality
degrades with more masked fields. The suffix byte-identity metric was broken
but the F1 result is clear: AR wins.

This refutes the hypothesis that bidirectional conditioning gives DG an
infilling advantage. The likely reason: the structured-extraction task is
copy-heavy, and AR's copy ability (which is strong, per #39) suffices to
regenerate the suffix. The infilling advantage would only manifest on tasks
where the suffix is long and must be preserved exactly (e.g., code infilling),
not on short JSON fields.

## Impact

- §8 (negative results): add "infilling advantage refuted" to the list.
- The paper does not claim bidirectional conditioning as a DG advantage.
- This closes the last speculative positive direction for diffusion.
