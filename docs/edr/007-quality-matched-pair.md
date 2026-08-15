# EDR-007: AR-1.5B is the quality-matched baseline, not AR-7B

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #58, #34, #63

## Hypothesis

Every prior deployment claim compared LLaDA-8B against AR-7B (a
size-matched but not quality-matched pair). On 9-field scoring, AR-1.5B
(0.8628) matches or beats both DG (0.8518) and LLaDA-8B. The honest
baseline for a deployment claim is the *smallest AR model that matches or
beats the dLLM's quality* — which is AR-1.5B, roughly a fifth of the
parameters.

## Design

Compute 9-field F1 for AR-1.5B from the per-field scores in
`results/vllm_serving_1.5b.json`. Compare to DG's 9-field F1 from the
anchor re-measurement. Compute the memory, cold-start, and energy
arithmetic for the quality-matched pair.

## Prediction

- **P1:** AR-1.5B is smaller, faster to start, and lower-energy than every
  available dLLM configuration. The deployment argument is dead.
- **P5 (only possible survivor):** at long context and batch 1, the dLLM's
  absence of a growing KV cache may give it a flatter memory curve.

## Result

| Model | 9-field F1 | Weights (bf16) | vs AR-1.5B |
|---|---|---|---|
| AR-1.5B | 0.8628 | 3.0 GB | 1× |
| DG (K=8) | 0.8518 | 52.0 GB | 17.3× |
| LLaDA-8B | ~0.78* | 16.0 GB | 5.3× |

*LLaDA 12-field F1; 9-field not yet computed.

AR-1.5B matches or beats both dLLMs on quality while being 5–17× smaller.
Cold start: AR-1.5B weight load ~1s (enforce_eager), vs LLaDA ~5.3s. The 71
GB figure from #49 was vLLM's pre-allocated KV pool, not model demand.

**Artifacts:** `results/vllm_serving_1.5b.json` (per-field F1),
`results/anchors/anchor_remeasure.json` (DG 9-field F1)

## Decision

**Accepted:** the quality-matched pair is AR-1.5B vs DG/LLaDA-8B. The paper
does not claim a deployment advantage for diffusion. The negative result is
stated explicitly: "we looked here too, and there is no regime favouring
diffusion at the quality-matched pair."

P5 (long-context memory) is the only possible survivor and needs GPU
measurement. It is not blocking the backbone.

## Impact

- The paper's §6 (applications) has no surviving positive deployment claim
  for diffusion. This is a legitimate result, not a failure.
- The cost-law comparison (§4) now compares AR-1.5B alongside AR-3B,
  showing the law holds across model sizes.
- The decision procedure's validity domain states "NOT size-matched; the
  asymmetry runs in the diffusion arm's favour" — so the DG recommendation
  is conservative and the AR recommendation is not.
