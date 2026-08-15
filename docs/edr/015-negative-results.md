# EDR-015: The negative results — five refuted and three withdrawn

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #27, #24, #25, #38, #29, #16, #53, #48, #63

## Hypothesis

Several claims from the FlowFill paper and from early experiments in this
repo do not survive fair measurement on the pinned stack. The paper must
state what was refuted and why, not silently drop it.

## Design

For each hypothesis, run the experiment on the pinned stack with 9-field
scoring, first-object-wins parsing, and provenance. Record the result and
the reason for withdrawal.

## Prediction

Most early dLLM-favorable claims will not survive fair measurement. The
pattern will be: the dLLM advantage was produced by a harness defect
(decoding config, serving parity, parser, metric bug, or provenance error)
that biased against AR.

## Result

### Five refuted hypotheses

| Hypothesis | Refuted because | Issue |
|---|---|---|
| dLLM is more robust to perturbation | DM worse than AR in 10/10 categories | #27 |
| dLLM improves with iterative refinement | DM improves (+0.023) but AR improves too | #24 |
| dLLM beats AR under bounded deadlines | AR actually completes (metric bug fixed) | #25/#38 |
| Wide-schema cost scales as O(F²) for dLLM | F1 is flat, not quadratic | #29 |
| Quality is invariant to output length | Quality rises 0.31→0.85 then plateaus | #43 |

### Three withdrawn claims

| Claim | Withdrawn because | Issue |
|---|---|---|
| Drain-only matches LLM extraction | Drain F1 0.68 vs LLM 0.85 | #16 |
| Structural vs semantic field separation | Semantic fields have degenerate gold | #53 |
| M/D/1 queueing model predicts latency | Not predictive under bursty arrivals | #48 |

**Pattern confirmed:** each refutation traces to a harness defect that
inflated the dLLM's apparent advantage. The five defects (EDR-012) are the
mechanism behind the five refuted hypotheses.

## Decision

**Accepted:** the paper's §8 (negative results) states all eight explicitly.
Each carries a one-line reason and a linked issue. Reviewers reward this;
more importantly, it stops a future us from quietly reviving them.

## Impact

- The paper has zero surviving positive application claims for diffusion,
  except the narrow per-event latency regime (EDR-010 a/b/c).
- §8 is a legitimate section, not an apology. "We looked here too, and it
  doesn't work" is a finding.
- The negative results are the flip side of the checklist (EDR-012): the
  defects caused the false positives, and fixing the defects caused the
  refutations.
