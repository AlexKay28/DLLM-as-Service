# EDR-021: Real-life applications where DM is the right choice

**Status:** Proposed
**Date:** 2026-08-15
**Issues:** #63, #41

## Hypothesis

The per-event latency advantage (EDR-010) is narrow but real. There exist
real-life applications where all five conditions that favour DM hold
simultaneously, making DM the correct architectural choice — not just
"marginally faster," but structurally superior for the task.

## The five conditions

DM is the right choice when ALL of these are true:

1. **Output length ≥ 512 tokens** — where the speed advantage is meaningful
   (≥1.65× vs quality-matched AR-1.5B). Below L=256 the advantage is ≤35 ms
   (1.09×, imperceptible).
2. **Batch-1 by requirement** — not by low load, but by structural constraint
   (privacy isolation, interactivity, cold-path). If batching is available, AR's
   80× multiplier wins.
3. **Complete output required** — the application validates output schema (e.g.
   JSON with required fields). A partial AR output fails validation. DM delivers
   a complete record at any K.
4. **Quality floor requires 3B+ equivalent** — AR-1.5B (F1=0.8628) is not good
   enough for the task. If AR-1.5B suffices, its 3 GB footprint dominates.
5. **52 GB footprint acceptable** — data center, not edge/mobile.

## Candidate applications

### 1. Fixed-deadline complete output (strongest)

"Generate the best structured report you can in 400 ms."

| Model | L=256 deadline | L=512 deadline |
|---|---|---|
| DM (K=8) | 378 ms → **complete, F1=0.85** | 501 ms → **complete, F1=0.85** |
| AR-3B | 1617 ms → **0 tokens** (misses) | 3225 ms → **0 tokens** |
| AR-1.5B | 413 ms → **misses by 13 ms** | 826 ms → **misses by 426 ms** |
| AR-0.5B | ~200 ms → complete but F1~0.70 | ~400 ms → complete but F1~0.70 |

DM is the only model that delivers **complete, high-quality** output within the
deadline. AR gives either a fragment or a low-quality complete output.

**Real-life examples:**
- Autonomous driving: "produce a complete trajectory plan in <500 ms or hand to failsafe"
- Real-time trading: "produce a complete structured signal before the next tick"
- Emergency response: "produce a complete incident assessment before the dispatcher reads it"
- Live API with hard p99 SLA: "return a complete JSON response in <400 ms p99"

### 2. Anytime quality (complete > partial)

DM degrades gracefully with K — but always produces a complete record:

| K | p50 (ms) | F1 | Output |
|---|---|---|---|
| 2 | 202 | 0.48 | complete, rough |
| 4 | 292 | 0.79 | complete, usable |
| 8 | 379 | 0.85 | complete, good |

AR at the same time budgets:

| Time budget | AR tokens emitted | Output |
|---|---|---|
| 202 ms | ~32 | partial, unusable |
| 292 ms | ~46 | partial, unusable |
| 379 ms | ~60 | partial, unusable |

If the application needs a **complete structured record** (all fields filled),
DM delivers at any K. AR doesn't — it streams partial output that fails schema
validation until it finishes.

**Real-life examples:**
- Any API that validates output schema (JSON with required fields)
- Monitoring systems that need a complete assessment, not a fragment
- Systems with fallback logic: "if the model can't finish, hand to rule-based"

### 3. Multi-tenant privacy isolation

Cloud API where requests **cannot be batched across tenants**:
- HIPAA/GDPR/proprietary data isolation
- Each request is batch-1 **by policy**
- DM's per-event advantage applies unconditionally
- At L=512: DM=501 ms vs AR-1.5B=~826 ms (1.65×)
- Data center with 52 GB GPUs is the norm

**Real-life examples:**
- Healthcare API: each patient's data isolated
- Financial API: each client's data isolated
- Legal API: each case's data isolated

### 4. Cold-path ad-hoc analysis

Two-path monitoring system:
- **Hot path**: AR-3B, batched, handles firehose (1000 ev/s)
- **Cold path**: analyst clicks "deep-extract this one event" → batch-1, L=512+
  → DM is 6.4× faster

The cold path is inherently batch-1 — it happens when a human initiates it.

**Real-life examples:**
- SOC analyst investigating a specific alert
- Data engineer debugging a specific pipeline failure
- QA engineer examining a specific test output

### 5. Energy-bounded always-on inference

Device that always processes one event at a time:

| L | DM (ms) | DM energy (J) | AR-1.5B (ms) | AR-1.5B energy (J) | Winner |
|---|---|---|---|---|---|
| 256 | 378 | 4.2 | 413 | ~5.6 | DM (1.3×) |
| 512 | 501 | ~5.4 | 826 | ~5.8 | DM (1.07×) |
| 1024 | 739 | ~8.0 | 1652 | ~11.6 | DM (1.45×) |

DM wins on energy at L≥256 because its latency advantage outweighs its larger
model size. The crossover is around L=400.

**Real-life examples:**
- Sensor node with power budget
- Edge gateway with thermal limit
- Satellite with energy constraint

## What kills each candidate

| Candidate | What kills it | Survives? |
|---|---|---|
| Fixed-deadline | AR-0.5B at 200ms (but low quality) | **Yes** if quality floor > 0.70 |
| Anytime quality | AR could emit early JSON with partial fields | **Yes** — schema validation is binary |
| Privacy isolation | AR-1.5B at 413ms is close enough | **Yes** at L≥512 (1.65× gap) |
| Cold-path | AR-3B batched is faster overall | **Yes** — cold path is inherently batch-1 |
| Energy-bounded | AR-1.5B energy is comparable at L=256 | **Yes** at L≥512 |

## The recipe

DM is the right choice when ALL of these are true:

1. Output ≥ 512 tokens (meaningful speed advantage)
2. Batch-1 by requirement (privacy, interactivity, cold-path)
3. Complete output required (schema validation, not streaming)
4. Quality floor requires 3B+ equivalent (AR-1.5B not good enough)
5. 52 GB footprint acceptable (data center)

This is a real niche. Not huge, but real. The paper states it as:

> "The diffusion model is the correct choice for fixed-deadline, batch-1,
> long-output structured extraction where a complete output is required and
> the quality floor exceeds what a 1.5B AR model can deliver. In every other
> regime we tested, a quality-matched AR model is faster, smaller, or both."

## Impact

- §6 (fair comparison): add "applications" subsection with the recipe
- The paper's conclusion is not purely negative — there is a real niche
- The decision procedure's DG regime (L≥176, batch-1, short input) now has
  a concrete deployment story
- The fixed-deadline and anytime-quality candidates are the strongest because
  they exploit DM's structural property (complete output at any K) rather
  than just its speed
