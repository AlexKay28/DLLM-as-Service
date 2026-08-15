# EDR-013: Decision procedure as an implemented, tested artifact

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #64, #63

## Hypothesis

A decision procedure that takes (L, λ, batching, f1_tolerance) and returns
which architecture to serve on is the highest-value synthesis item in the
project. It turns five disconnected measurements (crossover, rate, batching
multiplier, quality price, canvas size) into one artifact a practitioner
can run.

## Design

Implement `models/decision/procedure.py` with five guards (truncation,
throughput, length, quality, default-DG). Every constant carries a
NUMBERS.md section reference and an artifact path. T_DG is injectable —
the one-variable form ships now; the two-variable form swaps in when #44
lands (it has). The step is a parameter; the CLI reports under min/max
published step.

Self-test (`test_procedure.py`) verifies:
- L_cross and λ_cross reproduced from law constants
- Fitted vs measured latency within tolerance
- All five guards fire in isolation
- All six worked examples produce expected arm
- Two-variable law mode emits warning
- Step-range robustness

## Prediction

11/11 tests pass. The procedure recommends DG only in a narrow regime
(batch-1, short input, L≥176, f1_tolerance ≥ 0.002).

## Result

11/11 tests pass. The DG regime exists but is narrow:

| Scenario | L | λ | Batch | Tol | Arm | Ratio |
|---|---|---|---|---|---|---|
| Interactive extraction | 176 | 0.5 | no | 0.05 | DG | 2.94× |
| Long summarisation | 2048 | 1.0 | no | 0.05 | DG | 10.65× |
| High-volume pipeline | 256 | 64 | yes | 0.05 | AR | — |
| Short field-completion | 32 | 0.5 | no | 0.10 | AR | — |
| Tight output budget | 128 | 0.5 | no | 0.05 | AR | — |
| Quality-critical | 512 | 1.0 | no | 0.01 | AR | — |

**Artifacts:** `models/decision/procedure.py`, `models/decision/cli.py`,
`models/decision/test_procedure.py` (11/11 passing)

## Decision

**Accepted:** the procedure is a contribution. It finds a real DG regime
(per-event latency) and correctly identifies that the regime is narrow. The
procedure is honest about its scope: it does not take deployment-budget
inputs, so it cannot find a DG deployment advantage (because none exists,
per EDR-007).

## Impact

- C15 goes green (procedure implemented, reproduces crossovers).
- The procedure's constants are calibrated with #44/#68 results (DG step
  updated from 118.83 to 119.6, input-length slope set to 0.123).
- The CLI runs from a clean checkout with no GPU.
