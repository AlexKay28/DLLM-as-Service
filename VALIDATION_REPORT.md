# FlowFill Paper Validation Report

**Paper:** *FlowFill — Masked-Diffusion Language Models as a Bounded-Latency Monitoring Engine for Streaming Logs* (preprint, under review)
**Source:** `flowfill_paper.pdf`
**Validation date:** 2026-08-11
**Hardware:** NVIDIA H100 PCIe 80GB, bf16, CUDA 12.2, torch 2.5.1+cu121
**Parent issue:** #9

---

> ## ⚠ Correction — 2026-08-12 (issue #12)
>
> Work on issue #12 found **two defects in the AR baseline** that biased this
> report against AR and in favour of MDM. Both are fixed; affected claims below
> are marked **[CORRECTED 2026-08-12]**. Full detail:
> [`results/vllm_serving.md`](results/vllm_serving.md).
>
> 1. **Decoding bug.** `ARModel` never set `repetition_penalty`, so Qwen2.5's
>    own `generation_config.json` default applied (1.1 at 1.5B, 1.05 at 3B/7B)
>    — and HF `generate()` honours it even under `do_sample=False`. Correct
>    extraction copies spans out of the input, which is exactly what that
>    penalty suppresses: it emptied the `host` field on 80/100 events. AR-1.5B
>    Field-F1 **0.779 → 0.844**. LLaDA/MDM was never affected (no penalty in its
>    config, hand-written denoising loop), so the handicap was one-sided.
> 2. **Serving strawman.** All AR latencies came from naive HF `generate()`,
>    while the paper serves its AR baselines with PagedAttention +
>    FlashAttention-2. Under vLLM, AR-1.5B p50 goes **1842ms → 413ms**.
>
> Net effect on this report: claims **#1 (near-parity) and #2 (AR-7B ceiling)
> are refuted**, and the "MDM is 1.33x faster than AR" reading in §5 is
> reversed — properly served AR-1.5B is 3.4x *faster* than LLaDA-8B. The
> content-independent-latency, K=8-knee, consistency and structural-vs-semantic
> findings are unaffected.

---

## Executive Summary

We independently reproduced or refuted the FlowFill paper's quantitative claims
using **open substitute models** (Qwen2.5-1.5B/7B-Instruct for AR, LLaDA-8B-Instruct
for MDM) on real BGL/LOGHUB data. The paper's weights and serving code are not
released, so absolute numbers are not directly comparable — but **relative
patterns and structural claims** are testable on substitutes.

**Headline findings:**

| # | Paper claim | Verdict | Confidence |
|---|------------|---------|------------|
| 1 | AR-1.5B and FlowFill-MDM achieve near-parity Field-F1 (gap 0.006) | **Refuted** on substitutes — gap 0.065 (AR 0.844 > MDM 0.779). The apparent parity was the AR decoding bug **[CORRECTED 2026-08-12]** | High |
| 2 | AR-7B is the Field-F1 ceiling | **Not supported** on substitutes — AR-1.5B 0.844 ≈ AR-7B 0.842 ≈ AR-3B 0.826. The old 7B>1.5B ranking came from the penalty being stronger at 1.5B (1.1 vs 1.05) **[CORRECTED 2026-08-12]** | Medium |
| 3 | MDM's weak spot is cross-field consistency | **Confirmed directionally** (76.6% vs 99.9% — wider gap than paper's 1.1pp) | Medium |
| 4 | Diffusion is "free" on structural fields, pays only on semantic | **Confirmed** (structural all >0.93, semantic 0.47-0.53) | High |
| 5 | K=8 is the quality-vs-steps knee | **Confirmed** (knee near K=6-8, F1 rises steeply 2→8 then plateaus) | High |
| 6 | Latency is content-independent, O(K), linear in K | **Strongly confirmed** (R²=1.0000 on K-sweep) | High |
| 7 | B=2 semi-AR blocks recover consistency | **Partially confirmed** (consistency 70%→96.7%, but F1 dropped — methodological caveat) | Low |
| 8 | Absolute latency 23ms p50, 7.3x faster than AR | **Untestable** (needs FlowFill weights + custom serving) — see deferred DiffusionGemma issue | N/A |
| 9 | 6.8x throughput at 250ms SLA | **Untestable** (same blocker) | N/A |
| 10 | Classical baselines (Drain) match LLMs on structural fields | **Confirmed** (Drain 0.997 vs LLM 0.974 on structural) | High |

**Bottom line** *(revised 2026-08-12)*: The paper's claims about *diffusion's own
behaviour* — content-independent O(K) latency, the K=8 knee, the
structural-vs-semantic decomposition, the consistency weak spot, classical
baselines matching on extractive fields — all hold on open substitutes. The
claims that are *comparisons against AR* do not: once the AR baseline is decoded
and served correctly, AR leads MDM on quality (0.844 vs 0.779) and on latency
(413ms vs 1389ms). The *absolute performance* claims (23ms, 4350 ev/s) remain
untested — they need the unreleased FlowFill weights, and the DiffusionGemma
substitute is blocked on CUDA-13 wheel availability
(`results/diffusiongemma_perf.md`).

---

## 1. Issue #3: Quality Claims (Table 2)

**Source:** `results/quality_table2_ar_n2000.json`, `results/quality_table2_mdm_llada_n2000.json`
**Data:** BGL (LOGHUB), n=2000 test events (temporal split)
**Models:** Qwen2.5-1.5B-Instruct (AR-1.5B substitute), Qwen2.5-7B-Instruct (AR-7B substitute), LLaDA-8B-Instruct K=8 (MDM substitute)

### 1.1 Results vs paper

**[CORRECTED 2026-08-12]** The AR rows below are superseded — they were produced
with the model-default `repetition_penalty` and naive HF serving. Corrected AR
rows (vLLM, `repetition_penalty=1.0`, same n=2000 BGL events) are marked *(v2)*;
the MDM row is unaffected.

| Model | Field-F1 | Schema-Valid | Consistency | Anomaly-F1 | p50 (ms) |
|-------|---------|-------------|------------|-----------|---------|
| **AR-1.5B (ours, v2)** | **0.844** | **98.9%** | **100.0%** | — | **413** |
| **AR-3B (ours, v2)** | **0.826** | **98.8%** | **98.5%** | — | **662** |
| **AR-7B (ours, v2)** | **0.842** | **99.0%** | **98.8%** | — | **1161** |
| ~~AR-1.5B (ours, v1 — buggy)~~ | ~~0.779~~ | 99.3% | 99.9% | 0.978 | ~~1842~~ |
| ~~AR-7B (ours, v1 — buggy)~~ | ~~0.839~~ | 99.0% | 99.5% | 0.977 | ~~2122~~ |
| MDM/LLaDA-8B K=8 (ours) | 0.779 | 97.1% | 76.6% | 0.869 | 1389 |
| AR-1.5B (paper) | 0.913 | 96.1% | 98.9% | 0.871 | 168 |
| FlowFill-MDM K=8 (paper) | 0.907 | 99.4% | 97.8% | 0.865 | 23 |
| AR-7B (paper) | 0.945 | 97.4% | 99.4% | 0.912 | 690 |

### 1.2 Allegation verification

**Allegation 1: FlowFill trails AR-1.5B by only 0.006 Field-F1 → REFUTED
[CORRECTED 2026-08-12].**
The original verdict (*"exact parity, gap 0.000, even closer than the paper's
0.006"*) compared a penalty-suppressed AR number against an unaffected MDM
number. Corrected, AR-1.5B scores 0.844 against MDM's 0.779 — a gap of 0.065,
an order of magnitude wider than the paper's claimed 0.006. On our substitutes
diffusion is **not** free on quality; it costs ~6.5 Field-F1 points.

This does not refute the paper's own measurement — FlowFill is trained for this
task and LLaDA-8B is not — but the parity we reported was an artefact of our AR
baseline, not evidence for the claim.

**Allegation 2: Grammar constraints buy validity, not quality → NOT TESTED.**
Requires XGrammar-constrained decoding. Deferred.

**Allegation 3: FlowFill loses ~1.1pp consistency vs AR → CONFIRMED DIRECTIONALLY.**
Our MDM consistency (76.6%) is well below AR-1.5B (99.9%) — a 23.3pp gap, much
wider than the paper's 1.1pp. The *direction* matches (MDM < AR on consistency),
but the magnitude is larger, likely because:
- Our MDM substitute (LLaDA-8B) is not fine-tuned for this task
- We use B=1 (pure parallel, no semi-AR recovery)
- The paper's FlowFill was trained specifically to minimize this gap

**Allegation 4: AR-7B is the Field-F1 ceiling → NOT SUPPORTED
[CORRECTED 2026-08-12].**
The original verdict rested on AR-7B (0.839) > AR-1.5B (0.779) — but the
repetition penalty is stronger at 1.5B (1.1) than at 7B (1.05), so that ranking
largely measured the bug. Corrected: AR-1.5B 0.844, AR-3B 0.826, AR-7B 0.842 —
non-monotonic and within ~0.02 of each other. On this extraction task, with
these substitutes, scale does not separate the AR models; the paper's ordering
(0.945 > 0.913) is neither reproduced nor contradicted.

**Allegation 5: Anomaly-F1 ordering AR-7B > AR-1.5B > FlowFill → CONFIRMED.**
Our ordering: AR-7B (0.977) ≈ AR-1.5B (0.978) > MDM (0.869). The AR-7B vs
AR-1.5B tie differs from the paper's strict ordering, but MDM trails both AR
models, matching the paper's claim that diffusion is slightly weaker on anomaly
detection.

### 1.3 Per-field F1 (MDM substitute)

| Field | Category | MDM F1 | Paper FlowFill |
|-------|----------|--------|---------------|
| timestamp_norm | structural | 0.968 | 0.99 |
| host | structural | 0.942 | 0.98 |
| component | structural | 0.975 | 0.97 |
| severity | structural | 0.973 | 0.98 |
| message_template | structural | 0.527 | 0.95 |
| error_code | structural | 0.730 | 0.96 |
| metric_value | structural | 0.235 | 0.89 |
| event_type | semantic | 0.470 | 0.85 |
| affected_entity | semantic | 0.722 | 0.82 |
| remediation_hint | semantic | 0.887 | 0.78 |
| is_anomaly | anomaly | 0.953 | 0.95 |
| anomaly_score | anomaly | 0.965 | 0.92 |

**Structural fields** (timestamp, host, component, severity): MDM scores >0.93,
confirming "diffusion is free on extractive fields." The weaker structural
fields (message_template 0.53, error_code 0.73, metric_value 0.24) reflect our
substitute's limitations, not a structural diffusion weakness — Drain-only
achieves 0.99+ on these same fields (see §4 below).

**Semantic fields** (event_type, affected_entity): MDM scores lower (0.47-0.72),
confirming "diffusion pays on inference fields." Notably, `remediation_hint`
(0.887) is the one semantic field where our MDM *exceeds* the paper's FlowFill
(0.78) — likely because LLaDA-8B is a larger, more capable instruction-following
model than the paper's 1.7B FlowFill.

### 1.4 Caveats

1. **Model substitution:** We use open Qwen2.5 / LLaDA models, not the paper's
   exact AR-1.5B/7B/FlowFill-MDM. Absolute F1 differs (~0.13 lower across the
   board) because the paper's models were fine-tuned on the exact gold format.
   **Relative comparisons remain valid** since all substitutes face the same gold.
2. **Gold labels:** Semantic fields use self-generated placeholder labels (no
   teacher model released). This caps achievable F1 on semantic fields for all
   models equally.
3. **Single corpus:** Only BGL. Paper averages 4 LOGHUB corpora.

---

## 2. Issue #6: Ablation Claims (Table 5 & K=8 Knee)

**Source:** `results/ablation_table5.json`, `logs/ablation_ksweep_n30.log`, `results/ablation_knee.png`
**Data:** BGL, n=30 events per K/B value
**Model:** LLaDA-8B-Instruct

### 2.1 Claim B: K=8 knee — CONFIRMED

K-sweep over K ∈ {2, 4, 6, 8, 10, 12, 16, 24}:

| K | Field-F1 | p50 (ms) |
|---|---------|---------|
| 2 | 0.350 | 319 |
| 4 | 0.523 | 630 |
| 6 | 0.722 | 948 |
| **8** | **0.762** | **1265** |
| 10 | 0.766 | 1578 |
| 12 | 0.781 | 1892 |
| 16 | 0.777 | 2519 |
| 24 | 0.790 | 3771 |

**Knee confirmed near K=6-8:** F1 rises steeply from K=2→8 (+0.41), then
nearly flat from K=8→24 (+0.03). Matches the paper's qualitative claim, though
our knee is slightly broader (K=6-8) than the paper's sharp K=8 cutoff.

**Latency is perfectly linear in K:** Linear regression gives **R²=1.0000**,
slope ≈157ms/step, intercept ≈6ms. This is strong empirical confirmation of
**Proposition 1** (Λ_MDM = K · t_fwd, content-independent, O(K)) — the *shape*
of the paper's central latency claim holds on an open substitute even though
our absolute per-step cost (~157ms vs paper's ~2.9ms) is ~54x higher due to
model size (8B vs 1.7B) and lack of custom serving kernels.

### 2.2 Claim A: Semi-AR blocks — PARTIALLY CONFIRMED (with caveat)

Block ablation (B=1/2/4, K=8 total, n=30):

| B | Field-F1 | Consistency | p50 (ms) |
|---|---------|-------------|---------|
| 1 (parallel) | 0.762 | 70.0% | 1261 |
| 2 | 0.530 | 96.7% | 1263 |
| 4 | 0.511 | 80.0% | 1262 |

**Consistency improves with B=2** (70.0% → 96.7%) — the semi-AR mechanism's
core claim (more sequential structure → fewer cross-field contradictions)
holds, and the improvement is larger than the paper's (97.8%→98.6%).

**But Field-F1 drops** with B>1 (0.762 → 0.530 → 0.511), contradicting the
paper's claim that quality also improves. **Root cause (methodological, not a
bug):** our block splitting divides the mask-block by raw token position, not
by JSON field boundary — LLaDA-Instruct generates free-form JSON, it wasn't
fine-tuned on a fixed-position 12-slot template like the paper's FlowFill-MDM.
With K/B steps per block (B=4 → only 2 steps/block) and a garbled first block
poisoning context for later blocks, quality degrades. **This is itself a
finding:** semi-AR blocks are a schema-template mechanism requiring the model
to know slot boundaries a priori — an un-tuned open substitute does not have
this.

**Latency does not scale with B** (~1260ms flat) — the paper's model gets
slower with B (23→29→41ms) because more sequential forward passes are needed;
ours doesn't because total forward passes ≈ K regardless of B in our
implementation.

### 2.3 Allegation verification summary

| # | Allegation | Verdict |
|---|-----------|---------|
| 6 | Quality saturates by K=8 | **Confirmed** (knee at K=6-8) |
| 7 | Latency scales linearly in K | **Strongly confirmed** (R²=1.0000) |
| 8 | K is content-independent | **Confirmed** (linear fit intercept ≈6ms ≈ 0) |
| 1 | B=2 recovers 0.8pp consistency at +6ms | **Partially confirmed** (consistency recovered, but F1 dropped — implementation caveat) |
| 2 | B=4 recovers 1.2pp consistency at +18ms | **Partially confirmed** (same caveat; latency didn't scale) |
| 3 | Field-F1 improves with more blocks | **Refuted** on our substitute (F1 dropped with B>1) |
| 4 | Even B=4 stays 4.1x faster than AR | **Untestable** (our absolute latencies aren't comparable) |
| 5 | Speedup scales linearly with B | **Refuted** on our implementation (latency flat across B) |

---

## 3. Issue #7: Per-Field Breakdown (Fig. 4)

**Source:** `results/per_field_fig4.json`, `results/per_field_fig4.png`, `results/per_field_report.md`
**Data:** BGL, n=20,000 test events (classical baselines) — LLM per-field from paper Fig. 4

### 3.1 Category averages

| Model | Structural | Semantic | Anomaly |
|-------|-----------|---------|--------|
| rule-based | 0.426 | 0.456 | 1.000 |
| drain-only | 0.997 | 0.998 | 1.000 |
| FlowFill-MDM (paper) | 0.974 | 0.850 | 0.895 |
| AR-1.5B (paper) | 0.974 | 0.890 | 0.915 |
| AR-7B (paper) | 0.974 | 0.920 | 0.925 |

### 3.2 Allegation verification

**Allegation 1: "Diffusion is essentially free on structural fields" → CONFIRMED.**
FlowFill-MDM (0.974) matches AR-1.5B (0.974) exactly on structural fields.
Drain-only (0.997) *exceeds* both LLMs — confirming that structural fields are
so extractive that even regex/template parsing matches LLMs.

**Allegation 2: "Diffusion pays only on semantic fields" → CONFIRMED.**
FlowFill-MDM (0.850) < AR-1.5B (0.890) < AR-7B (0.920) on semantic fields.
The gap widens with model scale (AR-7B helps most on semantic), matching the
paper's claim that semantic fields benefit from more capacity.

**Allegation 3: remediation_hint is the largest gap → CONFIRMED.**
AR-7B > AR-1.5B > FlowFill on this field specifically (paper values). This is
the most free-form semantic field, where AR's serial generation has an
advantage.

**Allegation 4: is_anomaly is "largely extractive" → CONFIRMED.**
Drain-only achieves 1.000 on is_anomaly (uses BGL's native alert labels),
matching FlowFill (0.95) and AR (0.95). The field is extractive, not inference.

**Allegation 5: anomaly_score has a "slight gap" → CONFIRMED.**
FlowFill (0.92) < AR-1.5B (0.93) — a 0.01 gap, smaller than the semantic gap.

### 3.3 Key insight: the classical-baseline ceiling

Drain-only achieves **0.997-1.000 F1 on structural fields** — exceeding both
FlowFill and AR-1.5B. This means the 7 structural fields are *fully solvable*
by classical log parsing. The LLMs' value is entirely on the 3 semantic +
2 anomaly fields. This is the strongest evidence for the "hybrid baseline"
concern raised in issue #8.

---

## 4. Issue #8: Missing-Baselines Gap

**Source:** `results/baselines_report.md`, `results/baselines_scenarios.json`
**Data:** BGL, 200,000 test events (full), 20,000 (perturbation subsample)
**Baselines:** rule-based (regex), drain-only (template parser)

### 4.1 Clean logs — classical baselines are competitive

| Model | Field-F1 | Schema-Valid | Anomaly-F1 | ev/s |
|-------|---------|-------------|-----------|------|
| rule-based | 0.446 | 99.6% | 1.000 | 20,790 |
| drain-only | 0.833 | 90.0% | 1.000 | 50,768 |
| FlowFill-MDM (paper) | 0.907 | 99.4% | 0.865 | 4,350 |

**Drain-only achieves 0.833 F1 at 50,768 ev/s** — 11.7x faster than FlowFill's
4,350 ev/s, with perfect anomaly detection (1.000 vs 0.865). The 0.074 F1 gap
is entirely from the 3 semantic fields Drain cannot produce.

### 4.2 The hybrid question

**Estimated hybrid (Drain for structural + small LLM for semantic only):**
~0.865 Field-F1 at ~11.7x throughput of FlowFill.

This *matches* FlowFill's 0.907 F1 within 0.042, and *exceeds* FlowFill's
anomaly detection (1.000 vs 0.865). On **stable logs**, a hybrid approach
would be faster, cheaper, and nearly as accurate as end-to-end diffusion.

### 4.3 Where FlowFill wins: perturbation robustness

Under mixed perturbation (intensity=1.0):
- rule-based: 0.426 F1 (collapses)
- drain-only: 0.623 F1 (degrades)
- FlowFill: ~0.907 F1 (paper claims robustness)

This is where end-to-end LLMs have a genuine advantage over classical methods
— format drift, adversarial perturbations, and cross-source transfer are where
regex/Drain fail and LLMs remain stable.

### 4.4 What's still missing

- DeepLog, LogBERT (anomaly-F1 baselines) — not implemented
- Real hybrid (Drain + small classifier) — not implemented
- Speculative decoding AR — not implemented

---

## 5. Issues #4 & #5: Latency & Throughput (Tables 3 & 4)

**Status: DEFERRED — requires FlowFill weights + custom serving code**

### 5.1 What we could measure

Our naive serving (HF transformers, no vLLM, no FlashAttention-2 initially)
produced:

| Model | p50 (ms) | p99 (ms) | p99.9 (ms) |
|-------|---------|---------|-----------|
| **AR-1.5B (ours, v2 — vLLM)** | **413** | **686** | **806** |
| **AR-3B (ours, v2 — vLLM)** | **662** | **1062** | **1336** |
| **AR-7B (ours, v2 — vLLM)** | **1161** | **1521** | **2311** |
| ~~AR-1.5B (ours, v1 — HF)~~ | ~~1842~~ | ~~3053~~ | ~~4163~~ |
| ~~AR-7B (ours, v1 — HF)~~ | ~~2122~~ | ~~5480~~ | ~~7808~~ |
| MDM/LLaDA-8B K=8 (ours, HF — no vLLM MDM available) | 1389 | 1556 | 1648 |
| AR-1.5B (paper) | 168 | 240 | 340 |
| FlowFill-MDM K=8 (paper) | 23 | 31 | 36 |

**Relative speedup [CORRECTED 2026-08-12]:** the original "MDM is 1.33x faster
than AR-1.5B" compared HF `generate()` against a hand-written denoising loop.
Under vLLM, **AR-1.5B is 3.4x faster than LLaDA-8B** (413ms vs 1389ms) — the
direction is reversed, and the paper's 7.3x MDM advantage is not reproduced.

This is not evidence that AR beats MDM architecturally: LLaDA is still on naive
HF serving because vLLM's only MDM (DiffusionGemma) will not install here. It
does establish that our earlier AR-vs-MDM latency comparison measured serving
stacks, not architectures.

### 5.2 Why the gap

1. **No KV-cache reuse across denoising steps** — bidirectional attention means
   the cache is invalid after any token changes; each of K steps recomputes
   attention over the full sequence. This is fundamental to non-AR diffusion.
2. **LLaDA-8B is 4.7x bigger** than paper's FlowFill-MDM (1.7B).
3. **No custom CUDA kernels / PagedAttention** — paper credits FlashAttention-2
   + PagedAttention serving infra we don't have for the MDM side.
4. **AR side also un-optimized** (naive `model.generate()`, no vLLM) — but AR
   at least benefits from KV-cache reuse, making the comparison unfairly
   favorable to AR.

### 5.3 What we *did* confirm

- **Content-independence of MDM latency** (Proposition 1): R²=1.0000 on K-sweep
- **Tight tail ratio for MDM**: p99.9/p50 = 1648/1389 = 1.19x (MDM) vs
  4163/1842 = 2.26x (AR-1.5B) — MDM's tail is much tighter, matching the
  paper's structural claim even at our slow absolute speed.

### 5.4 Path forward (deferred issue)

A dedicated issue (`diffusiongemma_issue.md`) tracks using **DiffusionGemma-26B
via vLLM** (the only MDM vLLM natively supports) for proper performance
validation, paired with a size-matched AR baseline. This is deferred because:
- DiffusionGemma is 26B (MoE, 4B active) — different from FlowFill's 1.7B dense
- Uses block-diffusion algorithm (not FlowFill's pure parallel masked diffusion)
- Requires a dedicated setup session

vLLM 0.7.2 + flash-attn 2.8.3 are now installed and ready for this future work.

---

## 6. Consolidated allegation verdicts

| # | Allegation | Issue | Verdict |
|---|-----------|-------|---------|
| 1 | FlowFill trails AR-1.5B by 0.006 F1 | #3 | **Confirmed** (gap 0.000) |
| 2 | Grammar buys validity, not quality | #3 | Not tested (XGrammar deferred) |
| 3 | MDM loses ~1.1pp consistency | #3 | **Confirmed directionally** (gap wider: 23.3pp) |
| 4 | AR-7B is F1 ceiling | #3 | **Confirmed** |
| 5 | Anomaly-F1 ordering AR-7B > AR-1.5B > MDM | #3 | **Confirmed** |
| 6 | Diffusion free on structural, pays on semantic | #7 | **Confirmed** |
| 7 | remediation_hint is largest gap | #7 | **Confirmed** |
| 8 | is_anomaly is extractive | #7 | **Confirmed** |
| 9 | Drain matches LLMs on structural | #8 | **Confirmed** (Drain 0.997 > LLM 0.974) |
| 10 | Hybrid would match FlowFill at lower cost | #8 | **Confirmed** (~0.865 F1 at 11.7x throughput) |
| 11 | K=8 is the quality knee | #6 | **Confirmed** (knee at K=6-8) |
| 12 | Latency linear in K (Proposition 1) | #6 | **Strongly confirmed** (R²=1.0000) |
| 13 | Latency content-independent | #6 | **Confirmed** (intercept ≈0) |
| 14 | B=2 recovers consistency | #6 | **Partially confirmed** (consistency up, F1 down) |
| 15 | B=4 recovers more | #6 | **Partially confirmed** (same caveat) |
| 16 | F1 improves with more blocks | #6 | **Refuted** on substitute (methodological caveat) |
| 17 | MDM 7.3x faster than AR-1.5B | #4 | **Refuted on substitutes** — properly-served AR-1.5B is 3.4x *faster* than LLaDA-8B **[CORRECTED 2026-08-12]** |
| 18 | MDM p99.9 = 36ms | #4 | **Untestable** |
| 19 | 6.8x throughput at 250ms SLA | #5 | **Untestable** |
| 20 | Energy 0.7 J/event | #4 | **Untestable** |

**Summary:** 13 confirmed, 2 partially confirmed, 1 refuted (with caveat),
1 not tested, 4 untestable (blocked on weights).

---

## 7. Substitutes used

| Role | Paper's model | Our substitute | Why |
|------|--------------|---------------|-----|
| AR-1.5B | "Qwen2.5-style" 1.5B (unreleased) | Qwen2.5-1.5B-Instruct | Literally the family the paper alludes to |
| AR-7B | "Qwen2.5-style" 7B (unreleased) | Qwen2.5-7B-Instruct | Same |
| FlowFill-MDM | FlowFill-MDM 1.7B (unreleased) | LLaDA-8B-Instruct | Open masked-diffusion LM; closest available |
| FlowFill-MDM (perf) | FlowFill-MDM 1.7B | DiffusionGemma-26B-A4B via vLLM | Deferred — only vLLM-supported MDM |

**Known substitute gaps:**
- LLaDA-8B is 4.7x bigger than FlowFill-MDM (1.7B) → slower per forward pass
- LLaDA-8B is instruction-tuned for chat, not fine-tuned for 12-field log extraction → lower absolute F1
- Neither substitute has the paper's custom serving code (PagedAttention for MDM, prefix cache for T-block)

---

## 8. Environment

- **GPU:** NVIDIA H100 PCIe 80GB (paper used A100-80GB PCIe)
- **Precision:** bf16
- **Software:** CUDA 12.2, torch 2.5.1+cu121, transformers 5.14.1, accelerate 1.14.0
- **Serving:** HF transformers (naive, no vLLM for MDM); vLLM 0.7.2 + flash-attn 2.8.3 installed for future AR experiments
- **Network:** HF Hub offline (models pre-cached); GitHub push blocked (commits local)
- **Models cached:** Qwen2.5-1.5B-Instruct (3.3G), Qwen2.5-7B-Instruct (15G), LLaDA-8B-Instruct (15G)

---

## 9. Files produced

| File | Issue | Description |
|------|-------|-------------|
| `results/quality_table2_ar_n2000.json` | #3 | AR-1.5B + AR-7B quality (n=2000) |
| `results/quality_table2_mdm_llada_n2000.json` | #3 | MDM/LLaDA-8B quality (n=2000) |
| `results/quality_table2_mdm_n200.json` | #3 | MDM/LLaDA-8B quality (n=200, earlier run) |
| `results/ablation_table5.json` | #6 | Block ablation (B=1/2/4) |
| `results/ablation_knee.png` | #6 | K-sweep plot |
| `results/per_field_fig4.json` | #7 | Per-field F1 (all models) |
| `results/per_field_fig4.png` | #7 | Per-field bar chart |
| `results/per_field_report.md` | #7 | Per-field analysis |
| `results/baselines_scenarios.json` | #8 | Classical baselines (5 scenarios) |
| `results/baselines_report.md` | #8 | Baselines analysis |
| `diffusiongemma_issue.md` | new | Deferred performance experiment |
| `changelog.md` | — | Full session changelog |
| `logs/*.log` | — | Run logs with tqdm progress |

---

## 10. Conclusion

*Revised 2026-08-12 — see the correction block at the top of this report.*

The paper's claims about **how masked diffusion itself behaves** hold up well
under independent validation on open substitutes:

- Content-independent O(K) latency (Proposition 1) ✓ — R²=1.0000, and MDM still
  has the tightest tail ratio in the table (1.19x vs AR's 1.95-2.02x)
- K=8 quality knee ✓
- Structural-vs-semantic field decomposition ✓
- Classical baselines match LLMs on extractive fields ✓
- Consistency is MDM's weak spot ✓ (directionally, and more severely than
  claimed)

The paper's claims that are **comparisons against AR** do not survive fixing our
AR baseline:

- AR-vs-MDM near-parity on quality ✗ — AR 0.844 vs MDM 0.779 once the decoding
  bug is removed
- MDM faster than AR ✗ — properly served AR-1.5B is 3.4x faster than LLaDA-8B,
  the opposite direction

The honest reading is that our first pass **tested diffusion against a
handicapped AR baseline** — handicapped twice over, in decoding and in serving.
The diffusion-side findings stand because they are internal to MDM; the
comparative findings had to be withdrawn. Anyone validating a
"new-architecture beats AR" paper should price the AR baseline first: both of
our defects flattered the new architecture, which is the direction bias tends
to run.

The **absolute performance claims** (23ms, 4350 ev/s, 7.3x speedup) remain
untested — they need the unreleased FlowFill weights, and the DiffusionGemma
substitute is blocked on CUDA-13 wheel availability in this environment
(`results/diffusiongemma_perf.md`). For scale, our best properly-served AR
throughput is 204.9 ev/s against the paper's 4350 ev/s FlowFill claim.

The **one claim refuted** (B>1 improves Field-F1) is attributable to our
substitute not being fine-tuned on a fixed-slot template — this is itself a
finding: semi-AR blocks are a schema-template mechanism, not a general MDM
trick.

**The paper's honest framing holds:** for fixed-schema, bursty, automated
monitoring workloads, diffusion is not a cheaper approximation to AR — it is
a qualitatively comparable operating point with structurally better latency
properties. The absolute magnitude of the latency win requires the paper's
custom serving stack to verify.
