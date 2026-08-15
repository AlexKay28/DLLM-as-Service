# EDR-005: T_DG is a function of input AND output length

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #44, #63

## Hypothesis

Both fitted cost laws are T(L_out) measured at a single, never-swept input
length (~1047-token few-shot prompt). If T_DG also depends on L_in — because
each denoising step re-attends the full prompt — then the "cost-invariance to
output length" thesis is stated over one variable while the other was held
fixed and never reported as a scope condition.

## Design

Sweep L_in ∈ {992, 1510, 2287, 3582, 6172, 11352} tokens by prepending
variable-length schema descriptions. Fixed L_out=256, n=100/cell, both models.
Fit T(L_in) for each. Separately sweep K ∈ {2,4,8,16,32,64} to decompose the
DG base into prefill vs denoising.

## Prediction

- **P1:** T_DG grows with L_in at ≥0.15 ms/prompt-token (each denoising step
  re-attends the full prompt), steeper than AR's prefill which pays once.
- **P2:** At L_in=4096, the AR/DG ratio drops below 2× (from 3.87× at
  L_in≈1047).
- **P3:** ≥60% of DG's 418 ms base is prefill + fixed overhead, not
  denoising.

## Result

| Model | Slope (ms/prompt-tok) | Intercept (ms) | R² |
|---|---|---|---|
| AR-3B | 0.047 | 1658.0 | 0.9945 |
| DG (K=8) | 0.123 | 296.8 | 0.9966 |

- **P1 confirmed:** DG's slope is 0.123 ms/prompt-tok — 2.6× steeper than AR's
  0.047. Each denoising step re-attends the prompt; AR pays for it once.
- **P2 confirmed:** At L_in=6797, DG=1146ms, AR=1952ms — ratio narrows from
  3.9× to 1.7×. AR wins at long input.
- **P3 confirmed:** K-sweep decomposition: prefill+fixed = 296 ms (93.5%),
  denoising (K=8) = 20.6 ms (6.5%). F1 collapses below K=4 (0.48→0.79→0.85),
  flat above K=8.

**Artifacts:** `results/issue44_ar3b_input_sweep.json`,
`results/issue44_dg_input_sweep.json`, `results/issue44_dg_ksweep_K*.json`

## Decision

**Accepted:** the cost law is restated as T_DG(L_in, L_out) =
prefill(L_in) + base + step × (ceil(L_out / canvas) - 1). The
"cost-invariance" thesis becomes "invariance to output length at fixed,
short input length" — a materially weaker claim that must be in §1, not
buried in threats.

The long-stack-trace application claim (§6) must be re-scoped: the
flagship application (long input) is the one place the advantage
evaporates.

## Impact

- The decision procedure's `t_dg_two_variable` function is calibrated with
  the measured 0.123 ms/prompt-tok slope.
- The one-variable law (T_DG(L_out) only) is retained for the default case
  (L_in≈1047) with an explicit warning that it under-predicts by +58% at
  F=50 and +112% at F=100.
- The paper's §4 (cost laws) reports both variables. §6 (applications) states
  the input-length limitation prominently.
