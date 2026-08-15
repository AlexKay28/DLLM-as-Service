# dLLM Findings — Consolidated Digest

**Date:** 2026-08-14
**Scope:** Everything this repo has measured about served masked/block-diffusion
LLMs vs autoregressive LLMs, distilled from `VALIDATION_REPORT.md`,
`DM_APPLICATIONS.md`, and the `results/` reports (issues #1–#27, all closed).
**Hardware:** A100/H100 80GB PCIe, BF16. **Substitutes:** Qwen2.5-{1.5,3,7}B (AR),
LLaDA-8B (HF + dInfer), DiffusionGemma-26B-A4B (vLLM 0.25.1).

---

## 1. The one-sentence answer

**If the workload can batch, AR wins; if it cannot, the diffusion model's
content-independent, output-length-invariant cost is the operative advantage —
and that advantage is architectural, measured on two independent runtimes.**

The paper we set out to validate (FlowFill) sold dLLMs as "low-latency log
engines, 7.3× faster than AR." That claim is **withdrawn**. What survived
scrutiny is a different, more durable claim built on two cost laws.

---

## 2. What held up from the paper (diffusion-internal claims)

| Claim | Result | Source |
|---|---|---|
| Latency is content-independent, O(K), linear in denoising steps K | R²=1.0000 on K-sweep | `results/ablation_table5.json` |
| K=8 is the quality knee | Knee at K=6–8 (F1 +0.41 to K=8, +0.03 after) | same |
| Diffusion is ~free on structural fields, pays on semantic | structural >0.93, semantic 0.47–0.72 | `results/quality_table2_mdm_llada_n2000.json` |
| Cross-field consistency is MDM's weak spot | 76.6% vs AR's 99.9% (wider than paper's 1.1pp) | same |
| Classical baselines match LLMs on extractive fields | Drain 0.997–1.000 on structural/anomaly | `results/per_field_fig4.json` |
| Semi-AR blocks recover consistency | 70%→96.7% at B=2 — **but** F1 collapses on an un-tuned model: semi-AR is a *schema-template* mechanism, not a general trick | `results/ablation_table5.json` |

## 3. What was withdrawn (AR-comparative claims)

Two defects in our own AR baseline had flattered diffusion; fixing them
reversed the headline findings (issue #12):

1. **Hostile decoding default** — Qwen2.5 ships `repetition_penalty` 1.1/1.05
   in `generation_config.json`, applied by HF `generate()` even under greedy
   decoding. Extraction *copies spans from the input*; the penalty emptied
   `host` on 80/100 events. Fix: AR-1.5B Field-F1 0.779 → **0.844**.
2. **Serving strawman** — naive HF loop vs paper's PagedAttention stack.
   Under vLLM: AR-1.5B p50 **1842 → 413 ms**, peak 204.9 ev/s.

Consequences: "AR≈MDM quality parity" → gap is 0.065 (0.844 vs 0.779);
"AR-7B is the ceiling" → AR scale is flat (0.844/0.826/0.842 across
1.5B/3B/7B — *extraction doesn't reward AR scale*); "MDM faster than AR" →
properly served AR-1.5B is 3.4× *faster* than LLaDA-8B. Issue #22 (LLaDA
under dInfer, a dLLM-native runtime) confirmed the withdrawal holds even with
the MDM side properly served.

**Meta-lesson:** when validating a "new-architecture beats AR" paper, price
the AR baseline first. Both our bugs flattered the new architecture — that is
the direction bias tends to run.

## 4. The measured cost laws (the backbone of everything)

A100, vLLM 0.25.1, serial serving (m=1), BGL (issue #17, `results/appD_report.md`):

```
T_AR(L) = 9.1 + 6.283·L                     ms   (linear, R²≈1.0, L to 2048)
T_DG(L) = 418.2 + 113.4·(⌈L/256⌉ − 1)       ms   (stepwise, canvas=256)
```

- **Marginal cost ratio ≈ 14–16×** (6.28 vs ~0.44 ms/token); DG advantage
  saturates ~14× as L→∞ and is **1.9× at L=128, 6.2× at L=512, 10.3× at L=2048**.
- **Length crossover L≈65 tokens**; **rate crossover λ≈1.8 ev/s** (AR-3B's
  knee is λ≈64–96 ev/s — batching absorbs 40× more load).
- Two orthogonal walls (#14): DG lag is flat in L but walls in λ; AR lag is
  flat in λ but walls in L.
- **Batch multiplier: DG 1.5–1.7× (vLLM), LLaDA 1.18× (dInfer), AR-3B 77–86×.**
  The ~1× dLLM cap held on a runtime with no vLLM bug → **architectural**:
  bidirectional attention is compute-bound at B=1, leaving no
  bandwidth-bound headroom for batching to convert. (#17 + #22 control)

## 5. Where dLLM wins — measured applications

| Application | Issue | Key result |
|---|---|---|
| **Long structured extraction** (stack traces, event cascades) | #17 | L=512: DG 519 ms vs AR 3226 ms (6.2×) **and** F1 0.795 vs 0.599 — AR forced to over-generate *rambles and corrupts even copy fields* (host 1.00→0.67). Diffusion is length-robust; AR is not. |
| **Serial investigation chains** (SOC triage, multi-hop forensic) | #21 | Concurrency=1 by construction removes AR's 77× multiplier; DG's single-request tail penalty *washes out under sequential composition* (p99.9 ratio →1.93× mean at depth 10–20, CLT). |
| **VIP/critical-alert preemption** | #15 | DG's ~420 ms service time is available on demand mid-stream inside a 1 s SLA where AR overshoots. |
| **Wide-schema normalization** (50–100 fields) | #19 | Consistency plateaus at 87% (kill condition not hit); joint-canvas denoising absorbs O(F²) inter-field constraints — structural coherence is a *decoding property*, not prompt engineering. |
| **Bounded-deadline completion** | #25 | DM record_completeness 100% at every deadline; AR **0%** at max_tokens=318 — truncated mid-JSON. The defensible form of the paper's "bounded latency." |
| **Iterative refinement / self-correction** | #24 | Re-masking only critic-flagged fields: **+0.023 F1**; AR must regenerate the whole record. |
| Edge / tenant-isolated / idle-mostly deployments | #17 | Extrapolated: B=1 DG *leads* (2.12 vs 1.18 ev/s); no queue ⇒ no AR multiplier. |

## 6. Where dLLM loses — honest boundary

- **Any batchable workload** — AR's multiplier is 50–80×; dLLM's is ~1×, capped architecturally (#17, #22).
- **High-rate streams** — Thunderbird at 211M lines: dLLM ≈1.6 years, Drain 69 minutes (#18). DM win region λ≲1.8 ev/s is below any real log stream.
- **Throughput trading** (#20) — windowing narrows the 18× deficit to 5.6×, still loses, pays lag per event.
- **Single-hop tail-sensitive** — DM p99.9/p50 = 3.05× vs AR 2.01× (#14); only under *composition* does the tail wash out (#21).
- **Perturbation/drift** — refuted; see §7.

## 7. Surprises worth publishing

1. **DM is *more* perturbation-sensitive than AR, in all 10 categories** (#27,
   hypothesis refuted): drops up to 8.2× AR's (SEPARATOR_CHANGE), and AR is
   *flat* under WHITESPACE/ABBREVIATION. Bidirectional conditioning appears to
   back-propagate input noise where left-to-right only forwards it. Mechanism
   diagnosis → issue #30/#31; size-matched disambiguation → #28/#29.
2. **DM F1 *improves* on novel events while AR degrades** (#23): DM 0.873→0.886
   routine→novel; AR 0.855→0.788 over the same gradient. Possibly genuine
   rarity robustness; binning-artifact check still open.
3. **AR never emits complete JSON under a deadline** (#25): 0% record
   completeness at every max_tokens value; field completeness caps at 8/12.
4. **The critic is the differentiator for masking-based editing** (#24 vs #26):
   one-shot imputation of low-confidence fields fails (P-F1 refuted, F1
   0.00–0.41), but the *same masking mechanism* gains +0.023 F1 when a critic
   selects the fields. Task framing > mechanism.
5. **Adaptive threshold decoding is DM's best quality number here** (#22):
   dInfer's threshold decoder lifts LLaDA-8B F1 0.779→**0.872**, consistency
   +11.7pp — adaptivity beats fixed-K on quality, at large latency cost.
6. **Semi-AR blocks need slot knowledge** (#6): B=2 recovers consistency but
   F1 collapses on a model not trained with known field boundaries — the
   paper's block trick is inseparable from its L/T/S schema training.

## 8. The final scoreboard

| Claim type | Outcome |
|---|---|
| Diffusion-internal (K-knee, O(K) latency, field decomposition, consistency gap, classical-baseline ceiling) | **Confirmed** |
| AR-comparative headline claims (parity, 7.3× speed) | **Refuted/withdrawn** after AR baseline fixes |
| Absolute numbers (23 ms, 4350 ev/s) | Untestable without FlowFill weights |
| 5 untested DM advantages (#23–#27) | 2 PASS (refinement, deadline), 1 MIXED (adaptive effort), 2 REFUTED (imputation, perturbation) |

## 9. Open threads (filed)

- **#28/#29** — size-matched revalidation of surviving advantages with
  DiffusionGemma-26B vs Qwen2.5-7B (every DM win to date used LLaDA-8B vs
  AR-3B; parameter asymmetry cuts either way).
- **#30/#31** — mechanism diagnosis of DM's perturbation sensitivity (10/10
  categories): intrinsic to bidirectional propagation, or LLaDA pretraining
  artifact?
- **dInfer cache fix** (`modeling_llada.py:938` errors on prompts ≥632 tok):
  would drop LLaDA-dInfer toward ~1389 ms @ F1=0.872 — the first candidate
  "DM wins both quality and speed vs AR-7B" data point.
- **vLLM MRV2 rewrite** (#52015) may lift DiffusionGemma batching; #22 says
  the ~1× cap is architectural, so: watch, don't block.

---

*Detail: `VALIDATION_REPORT.md` (paper claims), `DM_APPLICATIONS.md`
(applications + evidence map), `results/*.md` (per-issue reports).*
