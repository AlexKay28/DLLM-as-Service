# paper/BACKBONE.md

The skeleton of the new paper, organised around our own closed questions, not
FlowFill's original Tables 2–5. The FlowFill reproduction is one section
inside this paper, not its spine.

---

## 1. Thesis sentence

> Block-structured decoding (canvas-bound diffusion LLMs) and token-by-token
> decoding (autoregressive LLMs) have different cost shapes: a staircase with
> a prefill-dominated floor vs a line with no floor. The shapes produce a
> per-event crossover (L≈60 tokens, λ≈2.6 ev/s) below which the
> block-structured model is faster — but at the quality-matched pair, this
> crossover does not translate to a deployment advantage, because the
> quality-matched AR model is 5–17× smaller. The decision procedure
> formalises the crossover; the deployment envelope closes it.

### Paper identity (EDR-009, EDR-010)

A cost-law paper with a systems contribution. The contribution is the
decision procedure; the finding is that the per-event advantage is real but
narrow and does not survive the deployment comparison. The comparison pair
is **compute-matched** (AR-3B 3B active vs DG-26B-A4B 4B active, same quality)
for per-event claims, and **quality-matched** (AR-1.5B 3 GB vs DG 52 GB) for
deployment claims.

---

## 2. Claim ledger

The core artifact. Every claim the paper makes, one row each. Status is
green (verified against committed artifact on pinned stack, with spread
estimate), amber (measured but single-run or awaiting rescore), or red
(asserted but not established). **The issue does not close while any claim is
red.** Claims that cannot go green get cut from the paper, not softened.

| # | Claim | Section | Evidence artifact | Status | Verified by |
|---|---|---|---|---|---|
| C1 | AR and dLLM achieve comparable quality on 9-field extraction | §3 | `results/anchors/anchor_remeasure.json`, `results/issue43_nine_field_f1.json` | green | #57 anchors: AR-3B=0.8516, DG=0.8518 (spread<0.01) |
| C2 | The 12-field metric inflated AR's apparent advantage via 3 degenerate gold fields | §2 | `results/issue43_nine_field_f1.json` | green | 9-field rescore |
| C3 | AR cost is linear in output length: T_AR = a + b·L_out | §4 | `results/anchors/ar3b_anchor_summary.json` | green | #57 anchors (5×L=256 CV=0.05%, 3×L=512 CV=0.01%) |
| C4 | dLLM cost is a staircase in output length: T_DG = base + step·⌊L_out / canvas⌋ | §4 | `results/anchors/anchor_remeasure.json` | green | #57 anchors (DG L=256 p50=379ms CV=0.25%, L=512 p50=501ms CV=0.12%) |
| C5 | The DG base (418 ms) is K-dependent, not a constant | §4 | `results/anchors/anchor_remeasure.json` (K=8: 379ms), `results/dg_steps_32.json` (K=32: 416ms) | green | #57 anchors + discrepancy analysis |
| C6 | The 418→379 ms "base drift" is a K=32 vs K=8 configuration difference | §0 | `paper/NUMBERS.md` §0, `results/anchors/anchor_remeasure.json` | green | #57: K=8 base confirmed at 379ms (CV=0.25%, 5 repeats) |
| C7 | The AR-3B 1618→1915 ms drift is a vLLM 0.25.1 vs 0.7.2 stack difference | §0 | `paper/NUMBERS.md` §0, `results/anchors/ar3b_anchor_summary.json` | green | #57 anchors reproduce 1616 ms on 0.25.1 (CV=0.05%) |
| C8 | The per-canvas step cost is ~120 ms (not 100 ms) at K=8 | §4 | `results/issue68_dg_*.json` | green | #68: step=121.8/120.0/118.7/121.8/113.1/117.5 ms, median=119.6 ms |
| C9 | T_DG is also a function of input length (two-variable law) | §4 | `results/issue44_dg_input_sweep.json` | green | #44: DG slope=0.123 ms/prompt-tok vs AR 0.047 — DG is 2.6× steeper |
| C10 | The crossover is at L≈65 tokens | §5 | `models/decision/procedure.py:compute_l_cross` | amber | #64 (L*=59.1 at K=8 base 378.8; was 65.3 at K=32 base 418.24) |
| C11 | The crossover is at λ≈1.8 events/s | §5 | `models/decision/procedure.py:compute_lambda_cross` | amber | #64 (λ*=2.6 at K=8) |
| C12 | The dLLM batching multiplier is ~1× (stated as correlation, not mechanism) | §6 | `results/issue42_batching.json` | amber | #42 (retitle — mechanism not tested) |
| C17 | Canvas=128 is a free improvement (base -2.4%, F1 +0.011); canvas=64 is not viable | §6 | `results/issue66_canvas_{128,64}.json` | green | #66 |
| C18 | The staircase form generalises to HDFS (step ~120 ms on both corpora) | §4/§9 | `results/issue67_dg_hdfs_*.json` | green | #67: HDFS step=120.5 ms vs BGL 118.5 ms |
| C19 | AR cost law generalises to HDFS (AR HDFS p50 matches BGL within 0.2%) | §4/§9 | `results/issue67_ar3b_hdfs_*.json` | green | #67: AR HDFS L=256=1616.5ms vs BGL 1616.0ms |
| C13 | Four of five harness defects biased against the AR baseline | §2 | `CHECKLIST.md` | green | #65 |
| C14 | The five defects were all found by re-reading artifacts | §2 | `CHECKLIST.md` | green | #65 |
| C15 | The decision procedure reproduces the measured crossovers | §5 | `models/decision/test_procedure.py` (11/11) | green | #64 |
| C16 | DG's canvas_length=256 is a tunable config parameter | §6 | DG `config.json` | green | #66 step 0 |

---

## 3. Explicit non-claims

What the paper deliberately does not assert, and why.

### Five refuted hypotheses

| Hypothesis | Why refuted | Issue |
|---|---|---|
| dLLM is more robust to input perturbation | DM worse than AR in 10/10 categories | #27 |
| dLLM improves with iterative refinement | DM improves with selective masking (+0.023), but so does AR | #24 |
| dLLM beats AR under bounded deadlines | AR actually completes (metric bug fixed in #38) | #25/#38 |
| Wide-schema cost scales as O(F²) for dLLM | F1 is flat, not quadratic, in schema width | #29 |
| Quality is invariant to output length | Quality rises with L (0.31→0.85), then plateaus | #43 |

### Three withdrawn claims

| Claim | Why withdrawn | Issue |
|---|---|---|
| Drain-only matches LLM extraction | Drain F1 0.68 vs LLM 0.85 | #16 |
| Structural vs semantic field separation | Semantic fields have degenerate gold (#53) | #53 |
| M/D/1 queueing model predicts latency | M/D/1 not predictive under bursty arrivals | #48 |

### Claims killed or untested

| Claim | Status | Issue |
|---|---|---|
| The ~1× batching cap is *architectural* (KV-cache invalidation) | Retitled: stated as correlation, not mechanism | #42 |
| The first extra canvas is 16% cheaper | Dissolved: K=32 vs K=8 config difference | #57 |

---

## 4. The decision procedure

**Implemented:** `models/decision/procedure.py` + CLI (`python -m models.decision`)
+ self-test (`models/decision/test_procedure.py`, 11/11 passing).

Inputs: output length L, arrival rate λ, batching availability B, quality
tolerance Q.

Rule (first guard that fires wins):
1. **Truncation guard** — if L < natural_L_DG (176 tokens) → AR
2. **Throughput guard** — if batching=True and λ > λ* → AR
3. **Length guard** — if L < L* → AR (subsumed by truncation in current config)
4. **Quality guard** — if f1_tolerance < price_F1 → AR
5. Otherwise → DG, with T_AR(L)/T_DG(L) as expected speedup

Crossovers (from the law, not hardcoded):
- L* = 59.1 tokens (K=8 base) / 65.3 tokens (K=32 base)
- λ* = 2.6 ev/s (K=8) / 2.4 ev/s (K=32)

### Worked examples

| Scenario | L | λ | Batch | Tol | Arm | Ratio | Rule |
|---|---|---|---|---|---|---|---|
| Interactive single-record extraction | 176 | 0.5 | no | 0.05 | DG | 2.94x | default |
| Long stack-trace summarisation, batch-1 | 2048 | 1.0 | no | 0.05 | DG | 10.65x | default |
| High-volume log pipeline | 256 | 64 | yes | 0.05 | AR | 4.27x | throughput |
| Short field-completion | 32 | 0.5 | no | 0.10 | AR | 0.55x | truncation |
| Tight output budget, unknown record size | 128 | 0.5 | no | 0.05 | AR | 2.14x | truncation |
| Quality-critical extraction | 512 | 1.0 | no | 0.01 | AR | 6.49x | quality |

### Validity domain

- Hardware: one A100 80GB PCIe, Campaign B
- Runtime: vLLM 0.25.1, BF16, serial diffusion (--max-num-seqs 1)
- Models: DiffusionGemma-26B-A4B-it vs Qwen2.5-3B-Instruct (NOT size-matched)
- Task: 12-field structured extraction over BGL, few-shot ~1047 tokens
- MUST NOT be applied to: input lengths >> 1047 tokens (pending #44),
  batch-served diffusion, non-extractive tasks, canvas != 256, other
  GPUs/runtimes, or 12-field F1 tolerances (use 9-field)

---

## 5. Figure list

Each figure with its data source and the single claim it carries. Rebuilt
against the pinned stack (#57).

| Figure | Data source | Claim |
|---|---|---|
| F1 | Cost-law comparison: T_AR vs T_DG vs L | `results/p2_*.json` (re-measured) | C3, C4 |
| F2 | Canvas step edges: p50 at L=240,256,264,...,2048 | `results/issue68_dg_*.json` | C8 |
| F3 | Input-length sweep: T vs L_in for both models | `results/issue44_*.json` | C9 |
| F4 | Decision procedure: crossover diagram | Derived from C3,C4,C8,C9 | C10, C11 |
| F5 | Five defects: magnitude of each bias | `CHECKLIST.md` | C13, C14 |

---

## 6. Section map

Organised by our findings, not FlowFill's tables.

1. **Introduction** — thesis, contribution structure, the two comparison pairs
2. **Methodology: what we got wrong and how we fixed it** (EDR-001–003, secondary)
3. **The exchange rate: quality is comparable** (C1, C2 — compute-matched pair)
4. **The two cost laws** (C3, C4, C5, C6, C7, C8, C9 — two-variable, staircase, base decomposition)
5. **The decision procedure** (C10, C11, C15 — crossover, worked examples, validity domain)
6. **The fair comparison: where DM wins and where AR wins** (EDR-010 — a/b/c vs d/e/f)
7. **The batching boundary** (C12, C16 — correlation not mechanism, canvas tuning)
8. **Negative results** (§3 non-claims: five refuted + three withdrawn)
9. **Threats to validity** — single corpus (pending #67), single GPU, single model pair, two-variable scope

Working order: exchange rate → two cost laws → decision procedure → fair comparison → batching boundary → methodology → negative results → threats.
