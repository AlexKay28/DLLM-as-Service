# EDR-004: State the batching cap as correlation, not mechanism

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #42, #63

## Hypothesis

The paper's central mechanism sentence claims: "bidirectional attention
invalidates the KV-cache reuse that makes AR batching cheap, and leaves the
model compute-bound at B=1 with no bandwidth-bound headroom for batching to
convert." This causal link has never been tested. The ~1× dLLM batch
multiplier is inferred from a correlation (dLLMs have bidirectional
attention **and** a ~1× multiplier), not from a mechanism experiment.

## Design

Two options:
1. **Run the experiment** (Arm A: AR with cache disabled; Arm B: dLLM with
   cache enabled; Arm C: roofline at B=1). Cost: ~6–8 GPU-h.
2. **Retitle** — drop the mechanism assertion and state the correlation.

## Prediction

If we run Arm A and AR keeps a large multiplier with cache disabled (P1:
≥10× at B=32), the mechanism sentence is simply false. The true mechanism
would be arithmetic intensity, not cache reuse.

## Result

Not run. The #63 resolution-order comment explicitly offers retitle as a
free alternative: "If the mechanism experiment looks expensive, renaming the
paper after what was measured costs nothing and removes an untested
assertion from the title."

## Decision

**Accepted: retitle.** The batching cap is stated as:

> dLLMs exhibit a ~1× batch multiplier against AR's 77–86×. The cause is not
> isolated — candidates include bidirectional attention, MoE routing, model
> size, and kernel maturity.

The word "architectural" is removed from §1 and §4.3. The mechanism
experiment remains as future work.

## Impact

- The paper no longer depends on an untested causal claim.
- §5 (mechanism section) becomes a measurement section, not a mechanism
  section.
- If a future reviewer asks "why is the multiplier ~1×?", the answer is
  "we measured it; we did not isolate the cause" — which is honest.
- The retitle opens the door to a positive finding (canvas tuning, #66) as
  the paper's design contribution instead of the mechanism as its causal
  spine.
