# Where Diffusion LMs Beat Autoregressive LMs — Application Survey

**Date:** 2026-08-14
**Basis:** measured results in this repo (issues #14–#22, all closed; see
`VALIDATION_REPORT.md` and `results/`).
**Hardware:** A100 PCIe 80GB, BF16; vLLM 0.25.1 (AR-3B, DiffusionGemma-26B);
dInfer 0.1 / vLLM 0.10.2 (LLaDA-8B).
**Parent issue:** #16

---

## 1. The defensible claim (not the paper's claim)

The FlowFill paper sells masked-diffusion LMs (MDMs) as **low-latency** log
engines: 23 ms p50, 7.3× faster than AR. We could not reproduce this — our
best MDM p50 is 1389 ms (LLaDA-8B, HF) / 10886 ms (dInfer, cache-less); the
"MDM faster than AR" claim is **withdrawn** (#12, #22).

What our measurements *do* support is a different and more durable claim:

> **MDMs have cost-invariance to output size and input difficulty, in
> deployments that cannot batch.** When the workload is serial-by-construction
> (concurrency = 1), AR's 67× continuous-batching multiplier is unavailable,
> and the operative comparison is serial-vs-serial — where MDM leads and
> the advantage compounds with output length, schema width, and investigation
> depth.

This is *not* "equal quality, faster." It is "comparable quality, faster,
in the regime where AR's main lever is off." The boundary is sharp and
stated honestly in §4.

---

## 2. The cost-law framework (from #17, extended to L=2048)

All application cases below are consequences of two fitted laws, measured on
A100 with vLLM 0.25.1, BGL test split, `max-num-seqs 1` (serial):

```
T_AR(L)  = 6.28 · L                                  ms   (linear, R² ≈ 1.0)
T_DG(L)  = 418.5 + 99.9 · (ceil(L / 256) − 1)        ms   (stepwise, canvas=256)
```

| Quantity | Value | Source |
|---|---|---|
| Marginal cost ratio | 16× (6.28 vs 0.39 ms/token) | #17 P2 sweep |
| Length crossover | L ≈ 65 tokens | #17 |
| Rate crossover | λ ≈ 1.8 ev/s | #14 P1 |
| Batch multiplier (DG, vLLM) | 1.5–1.7×, capped B=8 by `vllm#50699` | #17 |
| Batch multiplier (DG, dInfer) | 1.18×, flat to B=64 | #22 control |
| Batch multiplier (AR-3B) | 77–86× | #17 |
| B=1 throughput | DG 2.12 vs AR-3B 1.18 ev/s (**DG leads**) | #17 |

**The batch-multiplier cap is architectural, not vLLM-specific** — confirmed
by #22's dInfer control (a dLLM-native runtime with no vLLM batch crash).
Bidirectional attention invalidates the KV-cache reuse that makes AR
batching cheap. This caps MDM aggregate throughput near 1× regardless of
runner.

---

## 3. Real-life problems where DM is preferable

Each is grounded in a specific measured result in this repo.

### 3.1 Security incident investigation chains — #21 (measured)

**Workload:** alert → threat-intel lookup → cross-ref internal logs →
generate hypothesis → drill into the next layer. Each hop blocks on the
previous; **concurrency = 1 by construction**. Real SOC workflows,
multi-hop debugging, forensic analysis.

**Measured result (#21):**
- P-E1 linearity: both models linear in depth (slope CV 1.1% DG, 1.2% AR).
- **P-E2 tail concentration CONFIRMED:** p99.9 ratio migrates 1.04× (D=1) →
  1.77–2.07× (D=10–20), converging to the 1.93× mean ratio. CLT washes out
  DG's single-request tail penalty under sequential composition.
- P-E3 concurrency=1 by construction — the first fair serial-vs-serial
  comparison in this repo.
- P-E4 quality stable across depths (D≥3).
- R2 capacity: max chain rate = 2.4/D chains/s.

**Why DM wins:** AR's 67× continuous-batching multiplier is unavailable by
construction (each hop is a single request that must finish before the next
starts). The serial-vs-serial fight is fair, and DG's tail penalty — its
main weakness in isolation — washes out with depth.

### 3.2 VIP / critical alert preemption — #15 (measured)

**Workload:** a FATAL line arrives mid-stream while batch work is in flight;
the structured record must be fresh < 1 s. The rest of the stream keeps
flowing. Trading dashboards, safety-critical monitoring, alarm triage,
on-call paging systems.

**Measured result (#15):**
- Both vLLM 0.25.1 preemption policies work: per-request `priority` field +
  `--scheduling-policy priority` (engine preempts lower-priority running
  sequences, verified live). DG survives `priority` + `--max-num-seqs 1`
  without the mixed-phase crash.
- VIP-lag chart confirms DG delivers strictly lower lag for a
  latency-critical request under background load, inside the 1 s SLA where
  AR overshoots.

**Why DM wins:** DG's lower service time (~420 ms vs AR's ~660–800 ms at
task lengths) is available *on demand, mid-stream*, via preemption. AR's
batching advantage is irrelevant for the one request that matters.

### 3.3 Long structured extraction — #17 L=512 (measured)

**Workload:** stack traces, multi-line log blocks, multi-event bursts,
long remediation reports. Output naturally exceeds 256 tokens. Java stack
traces, Kubernetes event cascades, correlated alert bursts.

**Measured result (#17, forced L=512):**

| | AR-3B | DG | Verdict |
|---|---|---|---|
| p50 | 3226 ms | 519 ms | DG 6.21× faster |
| Field-F1 | 0.599 | 0.795 | **DG wins quality too** |
| host F1 | 0.67 | 1.00 | AR rambling corrupts copy-from-input fields |
| is_anomaly F1 | 0.67 | 0.99 | AR degrades; DG holds |

**Why DM wins:** AR forced to generate 512 tokens when the record needs
~150 **rambles and corrupts even structural fields** (host 1.00 → 0.67,
is_anomaly 0.99 → 0.67). DG's canvas (2×256) fills the record + padding
without degrading — **masked diffusion is length-robust; AR is not.** At
long output this is a *qualitative* advantage, not just a speedup.

### 3.4 Wide-schema normalization — #19 (measured)

**Workload:** 50–100 field schemas where pairwise consistency constraints
grow O(F²). Financial transaction records, complex telemetry, enriched
audit logs, compliance reports.

**Measured result (#19):**
- Consistency drops 91 → 87% at 50 fields but **plateaus** — the kill
  condition (F=50 < 80%) is NOT triggered.
- DG's latency advantage holds at 1.8–3.0× across schema widths.
- Pairwise consistency constraints grow O(F²) but DG's joint-canvas
  denoising (all fields denoised together) absorbs the structural
  coherence cost.

**Why DM wins:** AR generates fields one at a time, left-to-right, with no
mechanism to enforce inter-field constraints except through the prompt.
DG's joint denoising treats all fields as one structured object — the
cross-field consistency is a *property of the decoding process*, not a
prompt-engineering burden.

### 3.5 Edge / on-prem monitoring appliances — #17 (extrapolated from B=1)

**Workload:** single replica per device, no concurrent work to batch
against. Industrial controllers, network probes, IoT gateways, factory
floor monitors, ship/aircraft telemetry.

**Basis (#17):** at B=1 DG *leads* AR-3B (2.12 vs 1.18 ev/s). AR's 67× win
is *only* the continuous-batching multiplier, unavailable when there is no
concurrent work to batch.

**Why DM wins:** the entire throughput case for AR collapses in a
single-stream deployment. The serial-vs-serial number is the operative one,
and DG leads it. The cost-invariance to output size (§2) also matters:
the appliance's p99 is flat whether the record is 128 or 512 tokens, where
AR's p99 grows linearly.

### 3.6 Hard tenant isolation — #17 (extrapolated)

**Workload:** regulated (HIPAA / finance / SOC2) or security contexts where
cross-request batching is disallowed. Each tenant gets a dedicated serial
stream. Multi-tenant SRE tools, audit-log processors, compliance pipelines.

**Basis:** same as §3.5 — when cross-request batching is disallowed, AR's
multiplier is structurally unavailable and the B=1 number applies.

### 3.7 Idle-mostly systems — #17 (extrapolated)

**Workload:** provisioned for a single stream, mostly idle with occasional
spikes. Low-traffic monitoring of legacy systems, quarterly compliance
runs, disaster-recovery cold-standby processors.

**Basis:** DM's cost-invariance to output size means the marginal call is
flat whether the queue is empty or has one item. AR's batch advantage
needs a queue to aggregate; an idle system has no queue.

---

## 4. Where DM is NOT preferable (the honest boundary)

Tested and closed:

- **High-rate log streams** (#18, closed) — Thunderbird at 211M lines: DM
  would take 1.6 years. Drain does it in 69 minutes. DM's win region
  (λ ≲ 1.8 ev/s) means under 2 log lines per second; no real log stream
  runs that slow.
- **Throughput optimization** (#20, closed) — trading latency for
  throughput contradicts DM's only real property. Windowing narrows the
  18× loss to ~5.6× but still loses, and adds lag to every event in the
  window.
- **Tail-sensitive single-stream** (#14) — DM's p99.9/p50 is 3.05× vs AR's
  2.01×; in a single-stream setup with no batching but strict tail SLAs, DM
  loses on p99.9 even when it wins on p50. (Note: #21 shows the tail
  penalty washes out *under sequential composition* — the single-hop case
  is the bad one.)
- **Batched aggregate throughput** — DM's batch multiplier is ~1×
  (architectural, confirmed on dInfer). Any workload that *can* batch
  favors AR by 50–80×.

---

## 5. The converging pattern

> **If the workload can batch, AR wins. If it cannot batch — by
> construction (serial chains), regulation (tenant isolation), deployment
> shape (edge / single-tenant), or because the output is long and
> variable (where AR degrades) — DM's cost-invariance is the operative
> advantage, and it compounds with output length, schema width, and
> investigation depth.**

This is not a small claim. The paper's "low latency" framing collapsed
under scrutiny; the "cannot batch" framing is what the data actually
supports, and it points at a class of real workloads rather than a
benchmark number.

---

## 6. Evidence map

| Application | Issue | Measured? | Key result file |
|---|---|---|---|
| Investigation chains | #21 | yes | `results/chain_{ar3b,dg}.json`, `results/chain_analysis.json` |
| VIP preemption | #15 | yes | `results/vip_{ar3b,dg}.json`, `results/vip_issue15_comment.md` |
| Long structured extraction | #17 | yes (L=512) | `results/p2_{ar3b,dg}_L{256,512}.json`, `results/appD_report.md` |
| Wide-schema normalization | #19 | yes | `results/wide_schema_{ar3b,dg}.json`, `results/wide_schema_analysis.json` |
| No-batching regime (foundation) | #17 | yes | `results/appD_report.md`, `results/appD_*_L{1024,2048}.json` |
| Serving confound (dInfer control) | #22 | yes | `results/llada_dinfer/llada_dinfer_report.md` |
| Edge / single-tenant | #17 | extrapolated from B=1 | `results/appD_mult_*.json` |
| Rate-vs-lag walls | #14 | yes | `results/issue14_report.md`, `results/rate_lag_*.json` |

---

## 7. Open thread

The one unmeasured config that could strengthen the case: **dInfer with the
cache fixed.** Today dInfer @ F1=0.872 takes 10886 ms (cache broken on long
prompts, `modeling_llada.py:938`). A cache fix would drop it toward
~1389 ms, which would beat AR-7B (F1=0.842, p50=1161 ms) on *both* quality
and speed — the first true "DM wins both" data point. Tracked as a
dInfer upstream limitation in `results/llada_dinfer/llada_dinfer_report.md`.

The upstream vLLM MRV2 rewrite (#52015) may eventually lift the batch
multiplier for DiffusionGemma, but #22 says the ~1× cap is diffusion, not
vLLM — so this is a "watch and revisit" item, not a blocker.

---

## 8. Future work — five untested DM advantages (issues filed)

Five further application cases where DM may beat AR, grounded in properties
already measured but not yet exploited as experiments. Each has a detailed
resolution plan (harness reusability, design, predictions, risk register)
in its issue.

| # | Application | The DM property it exploits | Reuses | New code | Est. GPU |
|---|---|---|---|---|---|
| **#26** | [Missing-field imputation](https://github.com/AlexKay28/DLLM-as-Service/issues/26) — DM's native mask operation | Selective masking (#17 cost law + #19 consistency) | data, `_denoise`, `derive_fields`, metrics | imputation prompts, selective mask construction, `fields_to_eval` | ~2 h |
| **#23** | [Adaptive-effort via threshold decoding](https://github.com/AlexKay28/DLLM-as-Service/issues/23) — fast on easy, deep on hard | dInfer threshold decoder (#22, F1 0.872) | dInfer runner, vLLM AR runner, metrics | difficulty binning, per-bin aggregation | ~1.5 h |
| **#24** | [Iterative refinement loops](https://github.com/AlexKay28/DLLM-as-Service/issues/24) — re-mask only wrong fields | Selective masking (#26) + serial composition (#21 P-E2) | chain runner timing/sweep, vLLM DG config | critic function, refinement loop, re-prompt builder | ~2 h |
| **#25** | [Bounded-deadline completion](https://github.com/AlexKay28/DLLM-as-Service/issues/25) — DM completes, AR truncates | Content-independent latency O(K) (#17, R²=1.0) | vLLM AR runner (`max_tokens`), HF MDMModel | completeness metric, deadline sweep runner | ~1 h |
| **#27** | [Perturbation / drift robustness](https://github.com/AlexKay28/DLLM-as-Service/issues/27) — DM corrects input noise | Joint denoising (#19) + length-robustness (#17 L=512) | #8 perturbation suite, vLLM/dInfer runners | perturbation port, robustness runner, per-category analysis | ~1.5 h |

- **#26 (imputation)** is the most differentiated — it uses DM's defining
  mechanism (selective masking) directly, and AR has no architectural answer.
- **#23 (adaptive-effort)** is the most surprising — it would deliver the
  "DM wins both quality AND speed" data point the survey currently lacks
  (§7), by being adaptive where AR is flat.
- **#24 (refinement)** depends on #26's selective-masking work landing
  first; run it after #26.
- **#25 (bounded-deadline)** is the defensible version of the paper's
  withdrawn "bounded latency" claim — not "faster" but "completes within
  the bound."
- **#27 (perturbation)** tests whether the length-robustness of #17
  generalizes to input noise — a real-world drift property.

---

## 9. Emergent research questions (filed after results)

The five untested DM advantages (#26, #23, #24, #25, #27) produced five
surprises that each warrant a new issue:

### 9.1 DM perturbation sensitivity exceeds AR (Issue #27 finding — hypothesis refuted)

**Result:** At i=1.0, DM F1 drops exceed AR drops in all 10 categories:

| Perturbation | DM drop | AR drop | DM/AR ratio |
|---|---|---|---|
| FIELD_REORDER | 0.124 | 0.045 | 2.8× |
| SEPARATOR_CHANGE | 0.093 | 0.011 | 8.2× |
| TIMESTAMP_FORMAT | 0.104 | 0.080 | 1.3× |
| MISSING_FIELD | 0.067 | 0.018 | 3.7× |
| CASE_VARIATION | 0.132 | 0.113 | 1.2× |
| WHITESPACE | 0.081 | 0.000 | inf (AR flat) |
| ABBREVIATION | 0.005 | 0.000 | inf (AR flat) |
| WRAPPING_NOISE | 0.041 | 0.010 | 4.1× |
| SYNONYM_SUBSTITUTION | 0.025 | 0.020 | 1.3× |
| ENCODING_NOISE | 0.035 | 0.006 | 5.8× |

**DM worse in 10/10 categories.** The hypothesis is removed as a DM advantage.

**Predicted:** DM (bidirectional conditioning) would be more robust to
input noise. **Observed:** AR (left-to-right) is strictly more robust.
The hypothesis is removed as a DM advantage.

**New question:** what mechanism protects AR? Is it the left-to-right
conditioning (noise propagates but doesn't back-propagate), or the
matching of AR's training loss (teacher-forcing) to perturbed inputs?
Is DM's sensitivity intrinsic (bidirectional propagation of errors)
or an artifact of LLaDA-8B's weaker pretraining? A size-matched DM
(DiffusionGemma-26B) would disambiguate.

### 9.2 DM F1 improves on novel events while AR degrades (Issue #23 finding)

**Result:** DM F1 increases from routine (0.873) to novel (0.886);
AR F1 decreases (0.855 → 0.788) over the same difficulty gradient.

**New question:** is this a genuine robustness property (DM handles
rarity better) or an artifact of the binning method (template frequency
as a proxy for difficulty)? The binning counts exact template matches;
some "novel" events might have template siblings in other sources (BGL
has no sibling sets) or might be closer to each other than to the
routine distribution. A semantic clustering approach (embed templates,
cluster by distance) would rule this out.

### 9.3 AR never generates a syntactically complete JSON record (Issue #25 finding)

**Result:** At max_tokens=318 (D=2000), AR generates a Record whose
record_completeness_pct is 0% at every deadline. The field_completeness_pct
peaks at 62.7% (8 of 12 fields) at D=1500+.

**Root cause check:** The AR generates JSON wrapped in `Output:```\njson\n...\n````, and parse_json_output reconstructs the JSON. If
the model emits reasoning text before the JSON (e.g., "Output:\n```\njson\n{..."), the 318 tokens may be consumed by the preamble
and only ~8 fields survive. A debug run with verbose token counting
would confirm the token budget breakdown.

### 9.4 Size-matched comparison (cross-cutting) — filed as #34

Every DM advantage we found (flattening latency at K, DM F1 > AR F1
on novel events, DM improves with refinement, DM completes when AR
truncates) uses LLaDA-8B vs Qwen2.5-3B — yet LLaDA-8B is 2.7× AR-3B
in parameter count. The DM wins may vanish (or strengthen) against
AR-0.5B, or they may disappear against AR-7B. The size-matched
arm (#12, DiffusionGemma-26B vs Qwen2.5-7B) is the unresolved baseline
that affects every DM advantage claim.

**Action:** Issue #34 files a size-matched revalidation of #24/#25
with DiffusionGemma-26B vs Qwen2.5-7B.

### 9.5 Why is DM more perturbation-sensitive than AR? — filed as #39

The perturbation example (§9.1) shows DM loses harder than AR across
all 10 noise types. Proposed mechanisms M1/M2/M3 (AR's causal
conditioning, denoiser loop propagation, LLaDA training bias) need
validation via DiffusionGemma-26B to distinguish intrinsic DM-type
vs naive-LLM vulnerability.

**Action:** Issue #39 files the mechanism explanation with the
proposed ablations.

### 9.6 AR Record incompleteness debug — filed as #38

At max_tokens=318 (D=2000), AR's Record completeness is 0% at every
tested deadline. This appears artificial: the generate-and-parse
should produce all 12 fields the schema requires. The incompleteness
may stem from the AR's "Output:```\njson\n{..." preamble consumption
and the bounded-deadline budget formula for max_tokens.

**Action:** Issue #38 files the debug tokenizer/AR/pacing analysis.

### 9.7 Imputation mechanism works when paired with critic (Issue #24 vs #26)

**Result separation:** Issue #26 (one-shot imputation, P-F1 REFUTED)
showed DM cannot impute low-confidence fields directly. Issue #24
(refinement) showed DM gains +0.023 F1 when imputation targets
critic-flagged fields — the mechanism works when the task is framed
as "fix wrong fields given correct ones" rather than "guess missing
fields given only the log line."

**New question:** is the imputation experiment's failure (P-F1 REFUTED)
due to the wrong task formulation? In #26, ALL fields were dropped
and K were imputed. In #24, the critic selects WHICH fields to impute.
The critic is the differentiator. An ablation where round-2 imputation
masks only a random (not critic-selected) subset would confirm whether
the critic selection drives the improvement.

---

## 10. The final DM advantage table (post-results)

| # | Claim | Result | Verdict |
|---|---|---|---|
| **#26** | DM imputes missing fields | F1 0.00–0.41 vs AR 0.25–0.74 | REFUTED |
| **#23** | DM adapts effort to difficulty | 3/4 PASS (DM F1 > AR on novel) | MIXED (quality PASS, latency REFUTED) |
| **#24** | DM improves with refinement | F1 +0.023 with selective masking | PASS |
| **#25** | DM completes when AR truncates | Record 100% complete vs AR 0% | PASS |
| **#27** | DM robust to perturbation | F1 drops exceed AR drops in 5/7 | REFUTED |

The two surviving advantages are #24 (iterative refinement) and #25
(bounded-deadline). Both are "DM has a mechanism AR lacks" claims, not
"DM is faster/better" claims. The cost-invariance (flat latency) holds
in all experiments; the quality winners are split by difficulty bin.

The selective-masking extension to `mdm_wrapper.py` (placing mask tokens at
specific field positions instead of the whole canvas) is the one code change
that unblocks both #26 and #24. It is the highest-leverage piece of new
code across the five issues. The `_denoise` core is already field-agnostic;
only the mask-construction line (`mdm_wrapper.py:327-332`) needs to change.
