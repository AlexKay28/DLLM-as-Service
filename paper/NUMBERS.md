# Verified metrics — authoritative source for the paper

**Rule: no number enters a `.tex` file unless it appears here.** Every entry
below was read directly from the `results/*.json` artifact named in its row, not
from a summary `.md`. Where a summary report disagrees with its own JSON, the
JSON wins and the discrepancy is flagged **[DISCREPANCY]**.

Regenerate/re-verify with `paper/verify_numbers.py` (see §10).

---

## 0. Hardware and software provenance (do not mix these up)

Three distinct experimental campaigns ran on **different hosts and stacks**. The
paper must state which applies to each table.

| Campaign | Issues | GPU | Runtime | Models |
|---|---|---|---|---|
| **A** — AR baseline correction | #12 | H100 PCIe 80GB, driver 580.65.06 (CUDA 13.0) | vLLM 0.19.1, torch 2.10.0+cu128 | Qwen2.5-1.5B/3B/7B |
| **B** — cost laws, walls, applications | #14, #15, #17, #19, #21 | A100 PCIe 80GB, CUDA-13 driver | vLLM 0.25.1, torch 2.11.0+cu130 | DiffusionGemma-26B-A4B-it, Qwen2.5-3B |
| **C** — dLLM-native serving control | #22, #23 | A100 PCIe 80GB | dInfer 0.1 + vLLM 0.10.2, torch 2.8.0+cu128, transformers 4.57.6 | LLaDA-8B-Instruct |
| **D** — original HF-transformers pass | #3–#8 | H100 PCIe 80GB | HF transformers 5.14.1, torch 2.5.1+cu121 | Qwen2.5-1.5B/7B, LLaDA-8B |

Data throughout: **BGL** (LOGHUB), temporal split, test partition. Precision BF16.

---

## 1. The AR baseline correction (§2 of the paper) — Campaign A

Source: `results/vllm_serving_{1.5b,3b,7b}.json`, `results/vllm_serving.md`.
n = 2000 BGL events, identical events to the Campaign-D runs (directly comparable).

### 1.1 Defect 1 — decoding

`ARModel` never set `repetition_penalty`, so Qwen2.5's `generation_config.json`
default applied: **1.1** at 1.5B, **1.05** at 3B and 7B. HF `generate()` honours
it under `do_sample=False`. Extraction copies spans from the input — precisely
what the penalty suppresses.

| Quantity | Value |
|---|---|
| `host` field emptied on | **80 / 100** events |
| AR-1.5B Field-F1, before → after | **0.779 → 0.844** |
| MDM affected? | **No** — no penalty in LLaDA config, hand-written denoising loop |

The handicap was therefore **one-sided**. This is the paper's central
methodological point.

### 1.2 Defect 2 — serving

| Model | HF p50 (ms) | vLLM p50 (ms) | speedup |
|---|---|---|---|
| AR-1.5B | 1841.6 | **413.3** | 4.46× |
| AR-7B | 2122.0 | **1161.1** | 1.83× |

### 1.3 Corrected streaming latency, m=1 (paper Table 1)

| Serving | Model | p50 | p99 | p99.9 | p99.9/p50 | Field-F1 |
|---|---|---|---|---|---|---|
| vLLM | AR-1.5B | 413.3 | 685.5 | 805.8 | 1.95× | **0.844** |
| vLLM | AR-3B | 662.0 | 1062.1 | 1336.3 | 2.02× | **0.826** |
| vLLM | AR-7B | 1161.1 | 1521.1 | 2310.9 | 1.99× | **0.842** |
| HF | AR-1.5B (v1, buggy) | 1841.6 | 3053.4 | 4162.5 | 2.26× | 0.779 |
| HF | AR-7B (v1, buggy) | 2122.0 | 5479.7 | 7808.0 | 3.68× | 0.839 |
| HF | LLaDA-8B, K=8 | 1389.0 | 1555.5 | 1648.2 | **1.19×** | 0.779 |
| *paper* | *FlowFill-MDM K=8* | *23.0* | *31.0* | *36.0* | *1.57×* | *0.907* |

**Three claim reversals follow:**

1. **Parity refuted.** AR-1.5B 0.844 vs LLaDA-8B 0.779 → gap **0.065**, vs the
   paper's claimed 0.006.
2. **AR scale is flat.** 0.844 / 0.826 / 0.842 at 1.5B / 3B / 7B — non-monotonic,
   spread 0.018. Extraction does not reward AR scale. The earlier 7B > 1.5B
   ranking measured the penalty being stronger at 1.5B (1.1 vs 1.05).
3. **Direction of the speed claim reverses.** Properly served AR-1.5B is
   **3.36×** faster than LLaDA-8B (413.3 vs 1389.0 ms), against the paper's
   claimed 7.3× in MDM's favour.

**Retained MDM result:** LLaDA-8B still has the tightest tail in the table
(p99.9/p50 = **1.19×** vs AR's 1.95–2.02×). Content-independence survives the
correction — it is an MDM-internal property, not a comparison.

### 1.4 AR throughput under continuous batching (Campaign A)

`results/vllm_serving_1.5b.json`. Burst of B requests; p99 = burst wall-clock.

| Batch | AR-1.5B p99 (ms) | ev/s | tok/s |
|---|---|---|---|
| 1 | 418.6 | 2.4 | 332 |
| 32 | 634.1 | 50.5 | 6871 |
| 128 | 1141.1 | 112.2 | 15348 |
| 512 | 2498.5 | **204.9** | 28250 |

Peak AR-1.5B: **204.9 ev/s**. AR-3B peak: **145.2 ev/s** at batch 512.
No batch size meets a 250 ms p99 SLA for either model.

---

## 2. Cost laws (§3) — Campaign B

Source: **`results/appD_analysis.json`** (authoritative), `results/appD_report.md`.
A100, vLLM 0.25.1, BF16, `--max-num-seqs 1` (serial), n=300/cell.

### 2.1 Fitted laws — **REFIT 2026-08-14, supersedes all prior versions**

```
T_AR(L) = 7.27 + 6.29 · L                         ms      resid σ = 2.53 ms
T_DG(L) = 418.24 + 118.83 · (⌈L/256⌉ − 1)         ms      canvas = 256
```

`T_DG`'s **base** is solid: the four single-canvas cells (L = 32/64/128/256)
measure 416.9 / 418.5 / 418.1 / 419.4 ms — a spread of 2.5 ms — with mean
**418.24**, matching `appD_analysis.json`'s `dg_fit.base_ms` exactly.

`T_DG`'s **step is not**. Four different values are in circulation and **none of
the three published ones is the least-squares fit to the measured data**:

| Source | step (ms) | P-D2 err @1024 | @2048 |
|---|---|---|---|
| `DM_APPLICATIONS.md` §2 | 99.9 | — | — |
| `appD_analysis.json` `P_D2_check` (implied) | 100.25 | +8.4% | +11.7% |
| `appD_analysis.json` `dg_fit.step_per_canvas_ms` | 113.43 | +2.7% | +3.2% |
| **Least squares on measured cells (authoritative)** | **118.83** | **+0.6%** | **+0.0%** |

> **[BLOCKER — self-inconsistency inside one artifact]** `appD_analysis.json`
> contradicts *itself*: its `dg_fit` publishes step 113.43, but its `P_D2_check`
> predictions (719.0 / 1120.0 ms) imply step 100.25. So the file's own
> "extrapolation validated" check does not validate the law the same file
> publishes. The reported ±8.4%/+11.7% errors are an artifact of that mismatch;
> against the correct fit the errors are +0.6%/+0.0%. **The extrapolation claim
> gets stronger, not weaker, once the fit is corrected** — but the fitting
> script must be re-run and the artifact regenerated before submission.

> **[FINDING — the single-step model is mis-specified]** The per-canvas cost is
> not constant. Implied step from each multi-canvas cell:
>
> | cell | canvases | implied step |
> |---|---|---|
> | L=512 | 2 | **101.1 ms** |
> | L=1024 | 4 | **120.3 ms** |
> | L=2048 | 8 | **118.9 ms** |
>
> The **first additional canvas is ~16% cheaper** than subsequent ones. A single
> global step is therefore an approximation, and quoting one to four significant
> figures overstates its precision. This is a real observation about block
> diffusion's cost structure, not just bookkeeping — plausibly the second canvas
> overlaps with prefill, or a fixed overhead is amortised differently at nc=2.
> **Either report the law with a stated ±20% uncertainty on the step, or model
> the first step separately.** Do not present `118.83` as a precise constant.

### 2.2 Derived quantities

| Quantity | Value | Derivation |
|---|---|---|
| AR marginal cost | **6.29 ms/token** | fitted slope |
| DG marginal cost | **0.464 ms/token** | 118.83 / 256 |
| Asymptotic ratio | **13.6×** | 6.29 / 0.464 |
| Length crossover | **L ≈ 65 tokens** | measured: ratio 0.98 at L=64, 1.94 at L=128 |
| Natural output, DG | **~176 tokens** | `wide_schema_analysis.json` → `P_C1_latency/dg[0]/mean_tokens` = 176.455 |
| Natural output, AR | **~123 tokens** | same file, `P_C1_latency/ar[0]/mean_tokens` = 122.855 |

> **[CORRECTED]** An earlier revision cited `chain_analysis.json` for the natural
> output length; that file has no `mean_tokens` key. The value lives in
> `wide_schema_analysis.json`. Note also that the two models emit **different**
> natural lengths for the same 12-field schema (176 vs 123) — so "the task's
> natural output" is model-dependent, and §3's regime argument should cite the
> range 123–176, not a single figure.

### 2.3 Measured latency and ratio by L (n=300/cell)

**The ratio column below is the measured quotient**, not the fitted ratio.

| L | T_AR p50 (ms) | T_DG p50 (ms) | AR/DG measured | AR/DG fitted |
|---|---|---|---|---|
| 32 | 210.6 | 416.9 | 0.51 (AR wins) | 0.50 |
| 64 | 411.5 | 418.5 | 0.98 | 0.98 |
| 128 | 812.6 | 418.1 | 1.94 | 1.94 |
| 256 | 1617.6 | 419.4 | 3.86 | 3.87 |
| 512 | 3226.4 | 519.3 | **6.21** | 6.07 |
| 1024 | 6443.3 | 779.1 | **8.27** | 8.50 |
| 2048 | 12892.7 | 1250.7 | **10.31** | 10.63 |

> **[CORRECTED]** `appD_analysis.json`'s `ratio_T_AR_over_T_DG` is computed from
> the *fitted* laws, but was previously presented here as measured. The two
> diverge by up to 3.0% at large L, and a reviewer dividing the two adjacent
> columns will notice. **Cite the measured column in the text.** `DLLM_FINDINGS.md`
> already quotes the measured set (6.2×, 10.3×).
>
> Separately, `appD_report.md` reports DG at L=512 as **532 ms** while
> `p2_dg_L512.json` measures **519.3 ms**; the fitted ratio 6.07 is consistent
> with 532, the measured 6.21 with 519.3. Two different runs of the same cell
> appear to exist. Resolve before submission.

**Extrapolation check (P-D2), against the corrected fit:** predicted 774.7 ms at
L=1024 (measured 779.1, **+0.6%**) and 1250.0 ms at L=2048 (measured 1250.7,
**+0.0%**). ✓ Well inside ±15%.
**Linearity check (P-D3):** AR slope 6.29 vs expected 6.28, linear to L=2048. ✓

---

## 3. **Quality cost-invariance** (§3/§6) — the strongest single result

Source: `results/p2_{dg,ar3b}_L{32,64,128,256,512}.json`,
`results/appD_{dg,ar3b}_L{1024,2048}.json`. n=300/cell, forced `min_tokens = L`.

> ## ⚠ **[BLOCKER — this result may be a harness artifact. Issue #43.]**
>
> **Do not build the paper's headline claim on this table until #43 resolves.**
> Every link of an artifact mechanism is confirmed in committed code:
>
> 1. **AR is forced past EOS.** Each cell sets `min_tokens == max_new_tokens == L`
>    and `mean_output_tokens` is *exactly* L. A model that finishes the record at
>    ~123 tokens must emit ~389 more at L=512.
> 2. **The parser spans every repeated record.** `models/baselines/ar_model.py:181-183`
>    takes `text[text.find("{") : text.rfind("}")+1]` — first brace to **last**
>    brace. Concatenated objects fail strict `json.loads`, falling through to the
>    regex fallback.
> 3. **The fallback is last-wins.** `ar_model.py:213-217` assigns
>    `result[key] = ...` inside a left-to-right `finditer`, so the **final, most
>    drifted** copy of each field is the one scored.
> 4. **The observed signature matches exactly.** A last-wins parser over drifted
>    repeats yields *high validity* with *collapsing field accuracy* — precisely
>    the anomaly here: AR at L=512 holds `schema_valid` 99.3% while `host` falls
>    1.00 → 0.673 and `anomaly_f1` falls to 0.3553 on a boolean copied from a
>    native BGL label, which should be immune to output length. A model genuinely
>    degrading would lose validity too.
> 5. **Unfalsifiable from artifacts.** No `*_preds.jsonl` was saved for any
>    L-sweep cell, so this cannot be checked without a re-run.
>
> If a first-object-wins parser restores AR to ~0.8 at L≥512, then this section
> **and** §6's long-structured-extraction application both collapse, leaving a
> latency cost law only. Note the symmetry: this would be a **third** harness
> defect biasing against AR, in a paper whose central methodological claim is
> "price the AR baseline first." The truncation-regime rows may be similarly
> suspect — both models score *identically* 0.3167 at L=32 and L=64, which looks
> like an empty-prediction floor rather than a measurement.

| L | DG F1 | DG valid% | AR F1 | AR valid% | regime |
|---|---|---|---|---|---|
| 32 | 0.3167 | 0.0 | 0.3167 | 0.0 | truncation — both fail |
| 64 | 0.3167 | 0.0 | 0.3167 | 0.0 | truncation — both fail |
| 128 | 0.3287 | 3.3 | **0.6544** | 66.7 | truncation — **AR degrades gracefully** |
| 256 | 0.7871 | 99.3 | **0.8263** | 99.3 | matched (~176 tok natural) |
| 512 | **0.7945** | 100.0 | 0.5991 | 99.3 | over-generation |
| 1024 | **0.7887** | 100.0 | 0.4947 | 99.7 | over-generation |
| 2048 | **0.7902** | 100.0 | 0.4727 | 99.3 | over-generation |

**Three regimes, and this is the paper's headline figure:**

1. **Truncation (L ≤ 128).** Both collapse. **AR degrades gracefully, DG does
   not** (0.654 vs 0.329 at L=128; validity 66.7% vs 3.3%). Left-to-right
   decoding emits a valid *prefix* of fields; joint denoising truncated
   mid-canvas emits nothing parseable. **This is an AR advantage and must be
   reported as such** — it is the mirror image of the over-generation result and
   the digests omit it entirely.
2. **Matched (L = 256).** AR modestly ahead, 0.826 vs 0.787.
3. **Over-generation (L ≥ 512).** DG is **flat**: 0.7945 / 0.7887 / 0.7902 across
   512 / 1024 / 2048, validity 100% throughout. AR decays **monotonically**:
   0.5991 / 0.4947 / 0.4727. A 4-point monotone trend at n=300/cell.

**Mechanism, per-field at L=512** (`p2_*_L512.json`): AR corrupts fields it
should copy verbatim.

| Field | AR L=512 | DG L=512 |
|---|---|---|
| `host` | 0.6733 | **1.0000** |
| `component` | 0.7633 | 0.9900 |
| `severity` | 0.6167 | 0.9933 |
| `timestamp_norm` | 0.6100 | 0.8367 |
| `is_anomaly` | 0.6733 | 0.9933 |
| anomaly_f1 (overall) | **0.3553** | **0.9846** |

AR forced to over-generate does not merely waste time — it **degrades
copy-from-input extraction**, which should be immune to output length. DG's
corresponding fields are near-ceiling.

**Claim to make:** diffusion's invariance is to output length in *both* cost and
*quality*; AR's is neither. **Claim NOT to make:** "diffusion is more robust" —
it is strictly worse in the truncation regime.

---

## 4. Two orthogonal walls (§4) — Campaign B

Source: `results/rate_lag_dg_seq1.json`, `results/rate_lag_ar3b.json`,
`results/rate_lag_{dg,ar3b}_L{64,128,256,512}.json`, `results/issue14_report.md`.
Open-loop Poisson driver, lag = arrival → finish, 200 s or 400 events per rate.

### 4.1 Rate sweep

| λ (ev/s) | DG p50 lag | DG p99 | AR-3B p50 lag | AR-3B p99 |
|---|---|---|---|---|
| 1 | 434 ms | 1167 ms | 725 ms | 1293 ms |
| 1.5 | 566 | 2703 | — | — |
| 2 | 810 | 3613 | — | — |
| 2.5 | 4633 | 10574 | — | — |
| 4 | 21320 | 46612 (8 abandoned) | 719 | 1267 |
| 8 | — | — | 733 | 1284 |
| 32 | — | — | 893 | 1525 |
| 64 | — | — | 1783 | 2980 |
| 96 | — | — | 2828 | 4763 |
| 128 | — | — | 3932 | 6617 |

| Quantity | Value |
|---|---|
| DG saturation knee | **λ ≈ 2.0–2.5 ev/s** |
| AR-3B saturation knee | **λ ≈ 64–96 ev/s** (≈ 40× DG's) |
| p50 rate crossover | **λ ≈ 1.8 ev/s** |
| M/D/1 prediction for DG | μ ≈ 2.4 from ~420 ms service time — **matches the measured knee** |

The queueing-theory prediction matching the measured knee is a strong result;
state it explicitly rather than as an aside.

### 4.2 The joint grid — **Figure 1**

Uniform L ∈ {64,128,256,512} × λ ∈ {1,2,4,8,16}, both models.
**Two orthogonal walls:** DG lag is flat in L, walls in λ. AR lag is flat in λ,
walls in L. **DG's win region: λ ≲ 2 AND L ≥ 128** (5.3× at λ=1, L=512; tie at
L=64). Data: `results/rate_lag_{dg,ar3b}_L*.json`. Chart: `results/joint_rate_lag.png`.

### 4.3 Serving caveat — must be disclosed in the caption

DiffusionGemma runs `--max-num-seqs 1`; AR-3B runs the default 256-way
scheduler. **This asymmetry is forced by an engine bug, not chosen:** vLLM
0.25.1 crashes at `diffusion_gemma.py:96` (`size 257 vs 256`) as soon as
`num_running_reqs=2` mixes denoising phases. Dead-engine evidence:
`results/rate_lag_dg_engine_crash.json`. Upstream `vllm-project/vllm#50699`;
fix #51994 "trades the crash for serialization"; no parallel denoising batching
planned pre-MRV2 (#52015). Our repro is posted as issue comment 5282345519.
Serialized serving is therefore DG's production-relevant mode.

---

## 5. The mechanism: batching multipliers (§5) — Campaigns B + C

Source: `results/appD_analysis.json`, `results/llada_dinfer/llada_dinfer_report.md`.

| Runtime | Model | L | B=1 ev/s | peak ev/s | multiplier |
|---|---|---|---|---|---|
| vLLM 0.25.1 | DiffusionGemma-26B | 128 | 2.451 | 4.110 | **1.7×** |
| vLLM 0.25.1 | DiffusionGemma-26B | 512 | 1.872 | 2.820 | **1.5×** |
| dInfer 0.1 | LLaDA-8B | — | — | flat to B=64 | **1.18×** |
| vLLM 0.25.1 | Qwen2.5-3B | 128 | 1.230 | 94.90 | **77.1×** |
| vLLM 0.25.1 | Qwen2.5-3B | 512 | 0.310 | 26.68 | **86.0×** |

**Batching-capability gap: ~45× at L=128** (77.1 / 1.7).

**Why this is architectural, not a vLLM artifact — the load-bearing control:**
the ~1× cap reproduces on **dInfer, a dLLM-native runtime with no vLLM batching
bug**, at 1.18× flat to B=64. Two independent runtimes, two model families
(DiffusionGemma MoE / LLaDA dense), same cap.

**Proposed mechanism:** bidirectional attention invalidates the KV-cache reuse
that makes AR batching cheap; the model is compute-bound at B=1, so there is no
bandwidth-bound headroom for batching to convert into throughput.

**Honesty requirement:** DG's multiplier was measured with batch capped at 8
(the crash above). The measured 1.5–1.7× is therefore a **lower bound** on
vLLM's ceiling. The dInfer control is what licenses the architectural claim; the
DG number alone would not. Say this in the text.

**Also at B=1, DG leads.** Use the traceable pair from the table above:
**2.451 vs 1.230 ev/s at L=128** (`appD_analysis.json`), i.e. DG leads by 1.99×.
AR's entire throughput case is the multiplier, which requires a queue to exist.

> **[CORRECTED]** An earlier revision quoted "2.12 vs 1.18 ev/s" citing
> `appD_analysis.json`. That file's B=1 cells are 2.451/1.230 (L=128) and
> 1.872/0.310 (L=512); the 2.12/1.18 pair is from `diffusiongemma_perf.json` at
> *natural* output length, a different experiment. Both are legitimate — but they
> must not be mixed, and the L=128 pair is the one consistent with the rest of
> this section. Note the two pairs disagree on the *margin* (1.99× vs 1.80×)
> because they use different output lengths.

---

## 6. Application results (§6) — Campaign B

### 6.1 Serial investigation chains — `results/chain_analysis.json`

Depths D ∈ {1, 3, 5, 10, 20}. Concurrency = 1 **by construction** (each hop is a
single `generate([prompt])` that must finish before the next starts).

| Metric | D=1 | D=3 | D=5 | D=10 | D=20 |
|---|---|---|---|---|---|
| DG per-hop p50 (ms) | 418.6 | 407.2 | 410.0 | 409.0 | 416.9 |
| AR per-hop p50 (ms) | 807.8 | 823.7 | 825.2 | 833.7 | 837.0 |
| p50 ratio AR/DG | 1.930 | 2.023 | 2.013 | 2.038 | 2.008 |
| **p99.9 ratio AR/DG** | **1.041** | 1.872 | 2.070 | 1.894 | **1.769** |
| DG Field-F1 | 0.783 | 0.4565 | 0.4554 | 0.4654 | 0.4598 |
| AR Field-F1 | 0.8258 | 0.4953 | 0.4962 | 0.5054 | 0.5009 |

- **P-E1:** per-hop latency constant in depth (slope CV **1.1%** DG, **1.2%** AR)
  ⇒ total latency linear in D. ✓
- **P-E2 (the result):** the p99.9 ratio migrates from **1.041 at D=1** (DG's
  tail penalty exactly cancels its mean advantage) toward the **1.93 mean p50
  ratio** by D≥3. CLT washes out DG's single-request tail penalty under
  sequential composition. Analytical CLT: DG σ=268.2 ms, AR σ=158.7 ms; predicted
  cumulative p99.9 at D=20 is 12072 ms (DG) vs 18935 ms (AR).
- **P-E4 [CORRECTION to `DM_APPLICATIONS.md`]:** quality is **not** "stable across
  depths." Both models drop sharply D=1 → D=3 (DG 0.783 → 0.457; AR 0.826 →
  0.495) and only then plateau. **AR is ~0.04 better at every depth.** The chain
  result is a *latency* result only. State this.

### 6.2 Wide-schema normalization — `results/wide_schema_analysis.json`

| Fields F | DG p50 (ms) | AR p50 (ms) | AR/DG | DG consist.% | AR consist.% | DG valid% |
|---|---|---|---|---|---|---|
| 12 | 417.0 | 737.6 | 1.77 | 91.0 | 97.0 | 97.0 |
| 25 | 589.3 | 1440.2 | 2.44 | 92.5 | 94.0 | 98.0 |
| 50 | 1036.6 | 3121.8 | 3.01 | 87.0 | 96.5 | 97.5 |
| 100 | 2142.2 | 6239.1 | 2.91 | 87.5 | 96.5 | 99.0 |

- Kill condition (DG consistency < 80% at F=50) **not triggered**; DG plateaus at
  87.0–87.5%, non-monotonic (rises 91.0 → 92.5 before falling).
- DG validity **rises** with F (97.0 → 99.0) while consistency falls — the two
  decouple.
- **Caveat that must be stated:** DG emits more tokens than AR at every width
  (F=100: 1406.4 vs 950.7 mean tokens), so part of the latency ratio is DG doing
  more work, and part of the consistency gap may be length-related. AR holds
  96.5% consistency at F=100 vs DG's 87.5% — **AR wins consistency at every
  width.** The DG claim here is latency + graceful degradation, *not* superior
  consistency.

### 6.3 Bounded-deadline completion — `results/bounded_deadline_report.md`

AR-3B (vLLM), `max_tokens = D / 6.28`:

| D (ms) | AR F1 | AR field% | AR record% | AR p50 |
|---|---|---|---|---|
| 200 | 0.319 | 0.0 | 0.0 | 203 |
| 600 | 0.319 | 0.0 | 0.0 | 604 |
| 800 | 0.613 | 36.9 | 0.0 | 805 |
| 1000 | 0.808 | 60.7 | 0.0 | 808 |
| 1500 | 0.821 | 62.7 | 0.0 | 808 |
| 2000 | 0.821 | 62.7 | **0.0** | 807 |

DM (LLaDA-8B, K=8, one canvas): F1 **0.732**, field completeness **100.0%**,
record completeness **100.0%**, p50 **1077 ms**.

- **P-F7 PASS:** DM record completeness 100% vs AR 0% at D=1500.
- **P-F8 PASS:** AR F1 collapses below D=1000 (0.319 at D=200).
- **[OPEN — do not overclaim]** AR's 0% at *every* deadline, including D=2000
  where it has 318 tokens and finishes in 807 ms, is suspicious. Suspected cause:
  an `Output:```json` preamble consuming budget. Filed as **#38**. Until #38
  closes, report this as "AR's record completeness was 0% under our
  generate-and-parse harness" with the caveat, **not** as a clean architectural
  win. AR F1 0.821 vs DM 0.732 also means AR is *better* on quality here.

### 6.4 Iterative refinement — `results/refinement_report.md`

| Model | R1 | R2 | R3 | gain |
|---|---|---|---|---|
| AR-3B | 0.822 | 0.821 | 0.821 | **−0.001** |
| DM-8B | 0.747 | 0.763 | 0.770 | **+0.023** |

P-F5 PASS (gain > 0.01), P-F6 PASS (AR flat), **P-F9 REFUTED**: DM after three
rounds (0.770) still trails AR after one (0.822). The claim is "DM has a
refinement mechanism AR lacks," **not** "DM ends up better."

### 6.5 VIP preemption — `results/vip_{dg,ar3b}.json`, `results/vip_issue15_comment.md`

Both vLLM 0.25.1 preemption policies verified live: per-request `priority` +
`--scheduling-policy priority`; engine preempts lower-priority running
sequences. DG survives `priority` + `--max-num-seqs 1` without the mixed-phase
crash. At λ=0.5 ev/s DG lag p50 = 391.0 ms, p99 = 1147.2 ms. DG's ~420 ms
service time is available on demand mid-stream inside a 1 s SLA where AR
overshoots.

---

## 7. Negative results (§7)

### 7.1 Perturbation robustness — **REFUTED, 10/10** — `results/perturbation_report.md`

Clean baseline: AR-3B **0.8268**, DM **0.7390** (AR +0.0878). At intensity 1.0:

| Category | AR F1 | DM F1 | AR drop | DM drop | ratio |
|---|---|---|---|---|---|
| whitespace | 0.827 | 0.658 | −0.000 | 0.081 | ∞ (AR flat) |
| abbreviation | 0.829 | 0.734 | −0.002 | 0.005 | ∞ (AR flat) |
| separator_change | 0.815 | 0.646 | 0.011 | 0.093 | **8.2×** |
| encoding_noise | 0.821 | 0.704 | 0.006 | 0.035 | 5.8× |
| wrapping_noise | 0.817 | 0.698 | 0.010 | 0.041 | 4.1× |
| missing_field | 0.809 | 0.672 | 0.018 | 0.067 | 3.7× |
| field_reorder | 0.782 | 0.615 | 0.045 | 0.124 | 2.8× |
| timestamp_format | 0.746 | 0.635 | 0.080 | 0.104 | 1.3× |
| synonym_substitution | 0.807 | 0.714 | 0.020 | 0.025 | 1.3× |
| case_variation | 0.714 | 0.608 | 0.113 | 0.132 | 1.2× |

**DM worse in 10/10.** We predicted the opposite (joint denoising resists input
noise). Hypothesis: left-to-right conditioning propagates noise forward only,
while bidirectional conditioning also back-propagates it. Mechanism unresolved →
**#39**. Promote this to a full subsection; it is one of the paper's best results.

### 7.2 One-shot imputation — **REFUTED** — `results/imputation_report.md`

| K masked | AR F1 | DM F1 | Δ (DM−AR) |
|---|---|---|---|
| 1 | 0.250 | 0.000 | −0.250 |
| 2 | 0.591 | 0.413 | −0.178 |
| 4 | 0.743 | 0.301 | −0.442 |
| 6 | 0.643 | 0.154 | −0.489 |
| 8 | 0.688 | 0.221 | −0.467 |

P-F1 REFUTED; P-F2 PASS (DM latency flat in K, ratio 1.01); P-F3 REFUTED (AR
latency *also* flat, ratio 1.01); P-F4 REFUTED (cost ratio flat at 1.11× for all K).

**Pair this with §6.4:** the same masking mechanism fails at one-shot imputation
(F1 0.00–0.41) but gains +0.023 F1 when a **critic selects** the fields. Task
framing beat mechanism. This contrast is a genuine finding — give it a named
subsection.

### 7.3 Adaptive effort — MIXED — `results/adaptive_effort_report.md`

| Model | routine | medium | novel |
|---|---|---|---|
| AR-3B F1 | 0.855 | 0.833 | **0.788** |
| dInfer LLaDA F1 | 0.873 | 0.843 | **0.886** |
| AR-3B p50 (ms) | 826 | 808 | 807 |
| dInfer p50 (ms) | 10823 | 10995 | 11009 |

P-F1 PASS (DM 0.886 > AR 0.788 on novel); **P-F2 REFUTED** (DM p50 10823 ms vs
AR 826 ms — 13× slower); P-F3 PASS; P-F4 PASS (DM −0.013 vs AR +0.067 drop).
**Caveat:** difficulty bins use exact-template-frequency as a difficulty proxy;
binning-artifact check open. n=100/bin for DM vs 200/bin for AR.

### 7.4 Semi-AR blocks — Campaign D, `results/ablation_table5.json`

| B | Field-F1 | Consistency | p50 (ms) |
|---|---|---|---|
| 1 | 0.762 | 70.0% | 1261 |
| 2 | 0.530 | **96.7%** | 1263 |
| 4 | 0.511 | 80.0% | 1262 |

Consistency recovers at B=2 but F1 collapses, because our block split is by raw
token position, not field boundary — an un-tuned model has no slot priors.
Conclusion: semi-AR blocking is a **schema-template mechanism inseparable from
L/T/S training**, not a general MDM trick. n=30 only — label as preliminary.

### 7.5 MDM-internal claims that survive — Campaign D

| Claim | Result |
|---|---|
| Latency linear in K | **R² = 1.0000**, slope ≈157 ms/step, intercept ≈6 ms (LLaDA-8B) |
| K knee | K = 6–8 (F1 +0.41 from K=2→8; +0.03 from K=8→24) |
| Structural vs semantic | structural > 0.93, semantic 0.47–0.72 |
| Consistency is the weak spot | 76.6% vs AR 99.9% |

K-sweep: K=2 → F1 0.350 / 319 ms; K=8 → 0.762 / 1265 ms; K=24 → 0.790 / 3771 ms.

### 7.6 The classical-baseline ceiling — Campaign D, `results/baselines_report.md`

BGL, 200,000 test events. Drain-only: Field-F1 **0.833**, Anomaly-F1 **1.000**,
**50,768 ev/s**. Per-field, Drain reaches **0.997–1.000 on all 7 structural
fields** — matching or exceeding both LLMs (0.974). Rule-based: 0.446 F1.

Under mixed perturbation at intensity 1.0: rule-based **0.426**, Drain **0.623**
— classical methods degrade where LLMs hold.

Scale check (#18): Thunderbird at 211M lines — DM ≈ **1.6 years**, Drain
**69 minutes**.

**Implication to state plainly:** the LLM's value is confined to the 3 semantic
+ 2 anomaly fields. Any end-to-end extraction paper omitting Drain oversells.

> **[CAVEAT — must be disclosed]** `results/baselines_report.md` records
> *"Drain: logparser3 not installed — regex fallback used."* Our "Drain" is
> therefore **not the reference Drain implementation** but a regex approximation
> of it. This most likely *understates* the baseline, which makes the argument
> above conservative — but the label must not be left unqualified in the paper.
> Either install `logparser3` and re-run, or rename the row "Drain-style regex
> parser" throughout. Recommended: re-run; it is cheap and removes an easy
> reviewer objection against the paper's strongest baseline claim.

---

## 8. dInfer control (§5, §7.3) — Campaign C

Source: `results/llada_dinfer/llada_dinfer_report.md`. n=300 BGL, `threshold=0.9`,
`gen_len=256`, `block_length=32`, `cache=''`.

| Metric | HF LLaDA-8B K=8 | dInfer LLaDA-8B (threshold) |
|---|---|---|
| Field-F1 | 0.779 | **0.872** |
| Consistency | 76.6% | **88.3%** (+11.7 pp) |
| p50 | 1389 ms | **10886 ms** |

Adaptive threshold decoding is the best MDM quality number in the repo, at ~7.8×
the latency. **Blocker:** dInfer's prefix/vicinity KV-cache errors at
`modeling_llada.py:938` on prompts ≥632 tokens (`src [1,32,64,128]` vs slice
`[1,32,240,128]`), so all dInfer latencies are **cache-less**. dInfer's optimized
speed is unmeasured on this task. This is a dInfer limitation for long-prompt
serving, **not** a diffusion property — state that explicitly.

**The open opportunity:** a cache fix would move dInfer toward ~1389 ms at
F1 0.872, beating AR-7B (0.842, 1161 ms) on *both* quality and speed — the
"DM wins both" data point the paper currently lacks.

---

## 9. Threats to validity (§8) — state before a reviewer does

1. **Not size-matched.** Every DM result uses LLaDA-8B (8.02B) or
   DiffusionGemma-26B-A4B (26B total, 4B active) against Qwen2.5-3B. §§3–5
   survive this (cost laws are per-model measurements); **§6 does not**. → #34.
2. **Fit discrepancy** between `appD_analysis.json` and `appD_report.md` (§2.1).
   Must be reconciled before submission.
3. **DG serving asymmetry** forced by `vllm#50699` (§4.3).
4. **dInfer cache broken** ≥632-token prompts (§8).
5. **AR record-completeness 0%** may be a harness artifact → #38 (§6.3).
6. **Single corpus (BGL).** The FlowFill paper averages 4 LOGHUB corpora.
7. **Poisson arrivals understate log burstiness**; a bursty rerun moves both
   knees left. Lag past capacity is transient-windowed (300-event windows, 45 s
   drain), so the *knee location* is the result, not exact tail ms.
8. **Semantic gold labels are self-generated placeholders** (no teacher model
   released), capping semantic F1 for all models equally.
9. **FlowFill's absolute numbers are untested** — weights unreleased. Compare
   against its *claims*; never imply reproduction.
10. **Small n in places:** semi-AR ablation n=30; adaptive-effort n=100/bin (DM)
    vs 200/bin (AR).
11. **The headline quality result may be a parser artifact** (§3 warning box).
    → **#43**. Highest-priority open item; run before any other experiment.
12. **The batching mechanism is inferred, not measured.** The KV-cache
    explanation is consistent with the multipliers but was never tested
    directly, and the dInfer control that licenses the word "architectural" was
    itself run `cache=''` because of the `modeling_llada.py:938` bug — so a
    cache-based mechanism is argued from a cache-less measurement. → **#42**.
13. **`T_DG`'s step is not constant** (§2.1): 101.1 ms for the 2nd canvas vs
    120.3/118.9 for the 4th/8th. The single-step law is an approximation.
14. **"Drain" is a regex fallback**, not the reference implementation
    (`logparser3` was not installed) — §7.6 caveat. Re-run or relabel.
15. **Semantic gold labels are placeholder output of the same parser that
    produces structural gold**, so "semantic F1" may measure agreement with a
    regex rather than semantic quality — which would undercut the
    structural-vs-semantic decomposition in §7.6.
16. **The `consistency` metric is undefined in every report.** It is load-bearing
    for §6.2 and §7.6; its rule set has never been enumerated, and it is unknown
    whether the rule count scales with field count in the wide-schema
    experiment (which the O(F²) framing assumes).
17. **Chain quality collapses ~40% from depth 1 to depth 3** for *both* models,
    unexplained, before plateauing. §6.1's quality numbers are uninterpretable
    until the cause is known.
18. **Wide-schema latency is not like-for-like:** diffusion emits ~48% more
    tokens than AR at F=100 (1406 vs 951) for the same schema.
19. **No cell was ever repeated.** There are no confidence intervals anywhere in
    the repo, and at least one cell (DG at L=512) has two different recorded
    values across artifacts. → **#46**.

---

## 10. Numbers deliberately excluded

| Number | Why excluded |
|---|---|
| FlowFill 23 ms / 4350 ev/s / 7.3× | Unreproducible; weights unreleased. Cite as claims only. |
| "67× batching multiplier" (`DM_APPLICATIONS.md` §3.1) | Superseded by measured 77.1×/86.0× in `appD_analysis.json`. |
| `T_DG` step = 99.9 ms (`DM_APPLICATIONS.md` §2) | Superseded by 113.43 ms in `appD_analysis.json`. |
| "16× marginal ratio" | Superseded by 14.2× (6.29 / 0.443). |
| "P-E4 quality stable across depths" | Contradicted by the JSON: both models drop D=1 → D=3 (§6.1). |
| "DM robust to perturbation" | Refuted 10/10 (§7.1). |
| Energy J/event | Never measured; `avg_power_watts` was never captured. |

---

## 11. Verification

`paper/verify_numbers.py` re-reads every JSON cited above and asserts each value
in this file. Run before every submission build:

```bash
python3 paper/verify_numbers.py
```

Exit code 0 means every number in this file still matches its artifact.
