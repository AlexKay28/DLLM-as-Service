# EDR-022: Additional DM structural advantages — adaptive K and graceful degradation

**Status:** Proposed
**Date:** 2026-08-15
**Issues:** #63, #41

## Hypothesis

Beyond per-event latency, DM has two structural properties AR cannot match:
(1) a continuous per-request quality knob (K), and (2) graceful degradation
(quality degrades, completeness is preserved). These create real-life
applications that go beyond "DM is faster."

## Property 1: K as a per-request quality knob

### Finding

DM can adjust K per-request without model switching:

| K | p50 (ms) | F1 (9-field) | Throughput (ev/s) | Output |
|---|---|---|---|---|
| 8 | 379 | 0.853 | 2.6 | complete, good |
| 4 | 292 | 0.790 | 3.4 | complete, usable |
| 2 | 202 | 0.485 | 5.0 | complete, rough |

AR's equivalent is discrete: deploy a different model (0.5B/1.5B/3B/7B).
AR cannot adapt quality per-request without routing to a different model.

### Real-life application: multi-tier API

A single DM deployment serves three quality tiers:
- Premium requests (K=8, 379ms, F1=0.85)
- Standard requests (K=4, 292ms, F1=0.79)
- Economy requests (K=2, 202ms, F1=0.49)

The AR equivalent requires deploying 3 models (0.5B + 1.5B + 3B = 10 GB).
DM uses 1 model (52 GB) but never needs model switching. The trade-off:
AR is 5× cheaper in total memory; DM is simpler to operate (one model,
one serving path, per-request K).

### Real-life application: adaptive SLA compliance

If SLA tightens from 500ms to 400ms p99:
- DM: drop K from 8 to 4 → p99 ~380ms → **meets SLA**, quality drops 0.85→0.79
- AR-1.5B: p99 ~430ms → **misses SLA**, no knob to turn
- AR-3B: p99 ~1625ms → **misses SLA** badly

DM can meet a tighter SLA by sacrificing quality. AR cannot meet it at all
without deploying a smaller (lower-quality) model in advance.

## Property 2: Graceful degradation (complete > partial)

### Finding

When a deadline forces premature termination:

| Time budget | DM (K reduced) | AR (truncated) |
|---|---|---|
| 379 ms | K=8: F1=0.85, **complete** | ~60/256 tokens, **incomplete** |
| 292 ms | K=4: F1=0.79, **complete** | ~46/256 tokens, **incomplete** |
| 202 ms | K=2: F1=0.49, **complete** | ~32/256 tokens, **incomplete** |

DM degrades **quality**. AR degrades **completeness**. For schema-validated
outputs (JSON with required fields), this is a qualitative difference:
DM's output passes validation at any K; AR's truncated output fails.

### Real-life application: deadline-critical extraction

Systems that must produce a complete structured record within a hard deadline:
- Emergency response: "complete incident assessment in <400ms"
- Real-time trading: "complete structured signal before next tick"
- Safety-critical systems: "complete trajectory plan or hand to failsafe"

DM at K=4 (292ms, F1=0.79) delivers a complete, usable assessment. AR at
292ms delivers 46 tokens of a 256-token JSON — incomplete, fails validation,
unusable without a fallback parser.

### Real-life application: systems with fallback logic

Monitoring pipeline with three tiers:
1. **Primary**: DM K=8 (379ms, F1=0.85) — complete, good quality
2. **Degraded**: DM K=4 (292ms, F1=0.79) — complete, lower quality
3. **Fallback**: rule-based parser (instant, F1=0.68) — complete, lowest quality

AR's equivalent:
1. **Primary**: AR-3B (1616ms, F1=0.85) — complete, good quality
2. **Degraded**: AR-1.5B (413ms, F1=0.86) — complete, good quality (different model!)
3. **Fallback**: rule-based parser

AR's "degraded" path requires a second model deployment. DM's degraded path
is the same model with K=4.

## What the data also revealed (AR advantages)

### AR has tighter latency distribution

| Metric | AR-3B | DG |
|---|---|---|
| CV | 0.15% | 20.91% |
| IQR/median | 0.1% | 23.6% |
| Max/median | 1.03× | 3.26× |

AR's latency is essentially constant (content-independent under vLLM with
prefix caching and forced L=256). DG's latency varies 262% (range/median)
due to MoE routing variability and JIT warmup on the first events.

This is an AR advantage for capacity planning: tighter latency → smaller
headroom needed → more predictable resource allocation.

### Both models are deterministic

At temperature=0, both AR-3B and DG produce identical outputs across all 5
repeats (300/300 events). Determinism is not a differentiator.

## Summary: what's real and what's not

| Property | DM has it? | AR has it? | Real advantage? |
|---|---|---|---|
| Per-request quality knob (K) | Yes | No (discrete: model size) | **Yes** — simpler multi-tier |
| Graceful degradation (complete) | Yes | No (truncates) | **Yes** — schema validation |
| Tighter latency distribution | No (CV 21%) | Yes (CV 0.15%) | AR advantage |
| Content-independent latency | No | Yes (under vLLM) | AR advantage |
| Per-event speed at batch-1 | Yes (2–10×) | No | **Yes** (but narrow) |
| Determinism | Yes | Yes | Neither |

## Impact

- §6: add "adaptive K" and "graceful degradation" as DM structural properties
- These are the only DM advantages that don't collapse at the quality-matched pair
- The multi-tier API and deadline-critical extraction applications are real
- The paper's conclusion becomes: "DM's niche is not speed alone — it is the
  combination of per-event speed, per-request quality adaptivity, and
  guaranteed-complete output under deadline pressure."
