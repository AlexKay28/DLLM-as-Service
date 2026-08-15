# EDR-009: Paper identity — resolving the interrogation tensions

**Status:** Proposed
**Date:** 2026-08-15
**Issues:** #63, #41

## Your answers

| Q | Answer | Meaning |
|---|---|---|
| Q1 | a+b | Cost laws AND decision procedure are co-primary contributions |
| Q2 | a | Cost-shape difference caused by diffusion vs autoregression |
| Q3 | c | Two-variable finding is a §9 threat, not central |
| Q4 | b | Procedure always saying AR is a BUG — need to find DG's regime |
| Q5 | d | Checklist is an appendix, not load-bearing |
| Q6 | a | Full negative: "no deployment regime favouring diffusion" |
| Q7 | a | Base is 93% prefill → cost shape is structural (canvas), not algorithmic |
| Q8 | a | If HDFS fails, downgrade "law" to "observation" — paper still stands |
| Q9 | a+c | Systems paper + cost-law paper (dual framing) |
| Q10 | a | Title: "The Cost Shape of Diffusion: When It Wins and Why It Usually Doesn't" |

## Six tensions

### Tension 1: Q2a vs Q7a — what causes the shape?

**Q2 says:** the mechanism is diffusion vs autoregression (the decoding algorithm).
**Q7 says:** "the cost shape difference is structural (canvas) not algorithmic (diffusion)."

These contradict. The data (EDR-008) shows the denoising loop — the part that
makes it "diffusion" — is 6.5% of the cost. The staircase comes from the
canvas boundary, not from the diffusion process. A non-diffusion model with a
fixed output block size would have the same shape.

**Resolution:** the shape is caused by the canvas (structural), not the
decoding mechanism (algorithmic). The paper should say: "models that emit
output in fixed-size blocks (currently: block-diffusion models) have a
staircase cost; models that emit token-by-token (currently: AR models) have
a linear cost." The mechanism is the block structure, and diffusion is one
way to get block structure — but it is the block, not the diffusion, that
shapes the cost.

**Impact on Q10 title:** "The Cost Shape of Diffusion" overclaims if the
shape is about canvas, not diffusion. Consider: "The Cost Shape of
Block-Structured Decoding" or keeping "Diffusion" because it's the only
block-structured decoder we tested, with the scope stated explicitly.

### Tension 2: Q3c vs Q4b — where does the two-variable finding sit?

**Q3 says:** the two-variable law is a §9 threat, not central.
**Q4 says:** the procedure always saying AR is a bug — we need to find DG's regime.

But the two-variable law IS the explanation for why DG's advantage narrows at
long input. If it's buried in §9 as a threat, the paper can't honestly discuss
why the DG niche is narrow. The narrowing advantage is the two-variable law
in action.

**Resolution:** the two-variable finding must be at least a scope condition in
§1/§4, not a §9 threat. The decision procedure's one-variable mode carries a
warning that it under-predicts by +58% at long input — that warning is
load-bearing, not a footnote. Move from Q3c to Q3b.

### Tension 3: Q4b vs Q6a — is there a DG regime or not?

**Q4 says:** we need to find the regime where DG wins and make it the headline.
**Q6 says:** full negative — "no deployment regime favouring diffusion."

These contradict. If there's no regime (Q6), the procedure saying AR is
correct, not a bug. If the procedure is bugged (Q4), there must be a regime
we're missing.

**Resolution:** Q6 is the stronger answer (backed by data). The procedure is
NOT bugged — it correctly identifies that DG loses at the quality-matched pair.
The feeling that Q4b is true comes from the fact that the procedure is
currently incomplete: it doesn't take memory budget, cold-start budget, or
energy as inputs. Those are the axes where DG *might* win (though #58 P1/P2
suggest it won't). The resolution is:

- The procedure is correct for the inputs it takes (L, λ, batching, f1_tol).
- The procedure is incomplete — it needs deployment-budget inputs to be a
  full tool.
- The paper states: "on the inputs we measured, the procedure recommends AR
  in all tested scenarios. We do not claim this holds for deployment budgets
  we did not measure."

This turns Q4b from "the procedure is wrong" into "the procedure is honest
about its scope." The headline becomes: "the procedure is the contribution,
even when the answer is always AR — because the answer being always AR is
itself a finding."

### Tension 4: Q5d vs Q9ac — checklist as appendix in a systems paper?

**Q5 says:** checklist is an appendix.
**Q9 says:** systems paper (measurement framework + decision procedure).

A systems paper that claims a measurement framework as a contribution would
typically feature the methodology prominently. Burying the five defects (the
thing that validates the framework) in an appendix weakens the "systems"
framing.

**Resolution:** keep the checklist as a section (not appendix) but a
secondary one — §2 "methodology: what we got wrong and how we fixed it."
It supports the measurement framework claim (Q9a) without competing with the
cost laws (Q1a) for the headline. Move from Q5d to Q5b.

### Tension 5: Q8a vs Q10a — title vs generalization risk

**Q8 says:** if HDFS fails, downgrade "law" to "observation."
**Q10 says:** title is "The Cost Shape of Diffusion."

If the staircase is an observation (not a law), "The Cost Shape" overclaims.
The title presupposes the shape is real and general.

**Resolution:** the title is fine IF #67 confirms the form transfers. If it
doesn't, the title needs to narrow: "Cost Shape Observations for Block
Diffusion LLMs on BGL." Run #67, then decide. The current data (HDFS L=128,
256, 512) already shows the form holds (R²=0.96) — the risk is low.

### Tension 6: Q1ab vs Q6a — is the decision procedure a contribution if it always says AR?

**Q1 says:** the decision procedure is co-primary.
**Q6 says:** no deployment regime favours diffusion.

A decision procedure that always returns the same answer is not a decision
procedure — it's a proof. The paper needs to either:
- (a) Accept that the procedure is a proof (the answer is always AR at the
  quality-matched pair) and frame it as such, or
- (b) Show that the answer is NOT always AR — e.g., at batch-1, short output,
  short input, DG is faster (even though it's larger). The procedure's DG
  region exists; it's just narrow.

**Resolution:** (b) is correct. The procedure DOES recommend DG in some
scenarios (L≥176, batch-1, short input, f1_tolerance ≥ 0.002). The "always
AR" feeling comes from Q6's deployment framing, but the procedure's inputs
are about per-event latency and quality, not about total deployment cost. The
procedure says "DG is faster per-event at L≥176, batch-1, short input" — that
is a real regime, just not a deployment advantage because the quality-matched
AR model is also cheaper to deploy. The paper should state both:

1. The procedure finds a DG regime (batch-1, short input, L≥176): the per-event
   latency crossover is real.
2. The deployment envelope (memory, energy, cold start) does not favour DG at
   the quality-matched pair: the crossover is not a deployment advantage.

These are not contradictory — they are two different questions. The procedure
answers the first; #58 answers the second.

## Resolved answer set

| Q | Original | Resolved | Change | Reason |
|---|---|---|---|---|
| Q1 | a+b | a+b | — | Laws + procedure are co-primary |
| Q2 | a | **b** | Changed | Shape is from canvas, not diffusion. Data supports structural, not algorithmic. |
| Q3 | c | **b** | Changed | Two-variable must be scope condition in §1/§4, not §9 threat |
| Q4 | b | **a (modified)** | Changed | Procedure is correct, not bugged. It finds a narrow DG regime. "Always AR" is a deployment finding, not a procedure failure. |
| Q5 | d | **b** | Changed | Checklist is a secondary section, not appendix. Supports the systems-paper framing. |
| Q6 | a | a | — | Full negative on deployment. But the procedure still finds a per-event DG regime. |
| Q7 | a | a | — | Base is 93% prefill. Shape is structural. |
| Q8 | a | a | — | If HDFS fails, downgrade. Paper stands. |
| Q9 | a+c | a+c | — | Systems + cost-law dual framing |
| Q10 | a | a | — | Title holds IF #67 confirms. Current data is positive. |

## What this means for the paper

The paper is a **cost-law paper with a systems contribution**. The thesis:

> Block-structured decoding (currently: diffusion LLMs) and token-by-token
> decoding (currently: AR LLMs) have different cost shapes: a staircase with
> a prefill-dominated floor vs a line with no floor. The shapes produce a
> per-event crossover (L≈60 tokens) below which the block-structured model is
> faster. At the quality-matched pair, this crossover is real but does not
> translate to a deployment advantage, because the quality-matched AR model
> is 5–17× smaller. The decision procedure formalises the crossover; the
> deployment envelope closes it.

The paper's genre is: **"here is a real but narrow advantage, and here is why
it doesn't survive the deployment comparison."** The decision procedure is
the tool; the negative deployment result is the finding; the cost laws are
the evidence.
