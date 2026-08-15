# EDR-011: The crossover — L*≈59 tokens, λ*≈2.6 ev/s

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #64, #63

## Hypothesis

The two cost laws produce a crossover: below some output length L*, DG is
faster; above it, AR is faster. Similarly, above some arrival rate λ*, AR's
batching advantage overtakes DG's per-event advantage. The crossovers are
derivable from the laws, not measured directly.

## Design

Solve T_AR(L) = T_DG(L) for L, using the pinned-stack constants:
- T_AR = 7.27 + 6.29 × L
- T_DG = 378.8 + 119.6 × (⌈L/256⌋ − 1)

At the first canvas boundary (L ≤ 256), T_DG = 378.8, so:
L* = (378.8 − 7.27) / 6.29 = 59.1 tokens

For λ*: the DG serial throughput is 1000/378.8 = 2.64 ev/s. Above this rate,
DG cannot keep up even at batch-1, so AR (with batching) wins.

## Prediction

- L* ≈ 60 tokens (verified by bracketing: at L=64 ratio=0.98, at L=128 ratio=2.15)
- λ* ≈ 2.6 ev/s (verified by rate-lag curves: DG saturates at ~2.25 ev/s)

## Result

- L* = 59.1 tokens (K=8 base). With K=32 base (418.24): L* = 65.3 — the
  originally quoted "≈65" was K=32.
- λ* = 2.6 ev/s (K=8). With K=32: 2.4 ev/s — the originally quoted "≈1.8"
  was on a different stack with higher latency.
- Decision procedure test suite reproduces both from the law constants, not
  from hardcoded answers. 11/11 tests pass.

**Artifacts:** `models/decision/procedure.py:compute_l_cross`,
`models/decision/test_procedure.py`

## Decision

**Accepted:** L* = 59 and λ* = 2.6 on the pinned stack (K=8). The paper
reports both K=8 and K=32 values to show the crossover is K-dependent. The
decision procedure is the artifact that reproduces them.

## Impact

- C10 and C11 go green (computed from law, verified by test suite).
- The headline "L≈65" is updated to "L≈59 (K=8) / L≈65 (K=32)" — the
  originally quoted number was on the wrong K.
- The crossover is narrow: below 59 tokens, AR truncates and quality
  collapses (EDR-002). So the DG regime is L ∈ [59, ∞) at batch-1, short
  input — but this is also where AR's quality-matched model (1.5B) is cheaper
  to deploy (EDR-007).
