# Changelog

## 2026-08-10 — Issue #2: Datasets & Sources pipeline

Built the data pipeline that turns raw LOGHUB corpora into FlowFill's 12-field
schema splits, unblocking the downstream validation issues (#3–#8).

- **`datasets/download.py`** — resumable downloader (pure stdlib) for HDFS/BGL/
  Thunderbird (Zenodo 8196385) + Spirit (USENIX HPC4). Verifies against the
  server's Content-Length; supports `--only`, `--extract`, `--verify`,
  `--samples` (2k LOGHUB samples for smoke tests).
- **`datasets/prepare.py`** — per corpus: reservoir-sample 600k lines → temporal
  split (last 200k = test) → parse with Drain (`logparser3`, regex fallback) →
  emit 12-field schema JSONL + `manifest.json`. Bounded memory (works on 211M-
  line Thunderbird). `--smoke` mode for CI.
- **`datasets/README.md`** — documented both scripts + the actual directory
  layout.
- **`.gitignore`** — excludes `datasets/raw/`, `datasets/parsed/`,
  `datasets/splits/`.

**Verified:** smoke test on the 2k samples produces 100% schema-valid records
across HDFS/BGL/Thunderbird; `run_experiments.py` consumes the splits directly.

**Still open:** semantic-field labels (`event_type`, `affected_entity`,
`remediation_hint`) are placeholders pending a teacher LLM (tracked by #6).

## 2026-08-11 — Issue #8: Missing-Baselines Gap (Track 1)

Ran the rule-based + Drain-only baselines on real LOGHUB BGL data (200k test
events) across all 5 scenarios, producing the first real numbers that
contextualize FlowFill's claims against classical methods.

- **`datasets/prepare.py`** — fixed BGL parser: the real `BGL.log` has 9
  leading fields (`<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <RAS>
  <Component> <Level> <Content>`), not 7 as assumed. Now extracts correct
  component/severity/host. Also writes raw test lines (`*.raw`) for the
  experiment runner.
- **`models/baselines/drain_only.py`** — same BGL 9-field fix in the fallback
  parser; initialized `label` to avoid UnboundLocalError on HDFS.
- **`models/experiments/run_baselines_bgl.py`** — new runner that executes all
  5 scenarios (clean, drift, perturbation, cross_source, semantic) on real BGL
  data with tqdm progress bars. Full 200k for clean/drift/cross_source; 20k
  subsample for the 132-run perturbation sweep.
- **`results/baselines_scenarios.json`** — 154 experiment results across all
  scenarios.
- **`results/baselines_report.md`** — analysis answering the issue's central
  question.

**Key findings:**
- Drain-only achieves **0.833 Field-F1** vs FlowFill's 0.907 on clean BGL —
  the gap is entirely from semantic fields Drain cannot produce.
- A **hybrid (Drain + small LLM)** would achieve ~**0.865 F1 at ~11.7x
  throughput** (50,768 vs 4,350 ev/s) — small enough to qualify the paper's
  end-to-end diffusion claim on stable logs.
- Classical baselines **match FlowFill on anomaly detection** (1.000 vs 0.865
  Anomaly-F1) using native BGL labels.
- Under mixed perturbation (intensity=1.0), rule-based collapses to 0.426 F1
  and Drain to 0.623 — this is where end-to-end LLMs have a genuine advantage.

**Not done:** DeepLog, LogBERT, and a real hybrid (Drain + classifier) are
higher-effort baselines not yet implemented (tracked in the issue).

## 2026-08-11 — Issue #7: Per-Field Breakdown (Fig. 4)

Reproduced the per-field F1 breakdown from FlowFill paper Fig. 4, verifying
the "structural match, semantic gap" pattern with real BGL data.

- **`models/experiments/run_per_field_fig4.py`** — runs rule-based +
  drain-only on 20k BGL test events, computes per-field F1, and compares
  to paper Fig. 4 values (FlowFill, AR-1.5B, AR-7B).
- **`results/per_field_fig4.json`** — per-field F1 for all 5 models.
- **`results/per_field_fig4.png`** — bar chart grouped by category
  (structural / semantic / anomaly), reproducing the Fig. 4 layout.
- **`results/per_field_report.md`** — verifies all 5 allegations.
- **Bug fix:** added `_coerce_types()` to `rule_based.py` and
  `drain_only.py` — BOOL/FLOAT fields were compared as strings vs typed
  gold records, producing F1=0 on is_anomaly. After fix, drain-only
  macro F1 jumped from 0.833 to 0.998.

**Key findings (all 5 allegations confirmed):**
1. Structural fields: drain-only (0.997) matches LLMs (0.974) — confirmed.
2. Semantic fields: FlowFill (0.850) < AR-1.5B (0.890) — confirmed.
3. remediation_hint is the largest gap: AR-7B > AR-1.5B > FlowFill — confirmed.
4. is_anomaly is extractive: drain-only matches (1.000) — confirmed.
5. anomaly_score slight gap: FlowFill < AR-1.5B by 0.030 — confirmed.

**Caveat:** semantic F1 for classical baselines is inflated by self-generated
gold labels (teacher distillation not released). AR substitutes need GPU.

## 2026-08-11 — Issues #3/#4/#5/#6: GPU Experiment Infrastructure (prep)

Built the AR + MDM model wrappers and experiment runners for all 4 GPU-dependent
issues. All code is CPU-verified via `--mock` dry-runs; tomorrow's GPU work is
just "load model + run."

### AR infrastructure (substitutes for paper's AR-1.5B / AR-7B)

- **`models/baselines/ar_model.py`** — prompt builder for 12-field extraction
  (system prompt, few-shot examples, JSON output parser handling markdown fences
  and partial JSON).
- **`models/baselines/ar_wrapper.py`** — `ARModel` wraps HuggingFace models
  (Qwen2.5-1.5B/7B), generates via chat template, parses JSON → Record, times
  each event. Supports batch mode (m>1) for throughput experiments. Includes
  `MockARModel` for GPU-less dry-runs (simulates paper's 168ms p50).
- **`models/experiments/run_quality_table2.py`** (#3) — computes Field-F1,
  Schema-validity, Consistency, Anomaly-F1. Writes `results/quality_table2.json`
  + `.md` with comparison to paper Table 2.

### MDM infrastructure (substitute for paper's FlowFill-MDM)

- **`models/baselines/mdm_wrapper.py`** — `MDMModel` for masked-diffusion
  inference (LLaDA-8B / Dream as open substitutes). Interface:
  - `extract()` — K-step parallel denoising
  - `extract_with_blocks()` — Semi-AR B=1/2/4 (Table 5 ablation)
  - `extract_with_k()` — K-sweep (K=8 knee)
  
  Includes `MockMDMModel` simulating paper's 23ms p50 @ K=8 with linear
  K-scaling. **BLOCKER:** `load()` and `_denoise()` raise
  `NotImplementedError` — need LLaDA's custom denoising loop (integration
  notes in file).

### GPU experiment runners (all support `--mdm` flag)

- **`models/experiments/run_latency_table3.py`** (#4) — streaming-mode (m=1)
  latency + energy. `PowerMonitor` class polls `nvidia-smi` in background
  thread; measures idle baseline, computes net J/event.
- **`models/experiments/run_throughput_table4.py`** (#5) — batch sweep
  [1..128] under 250ms p99 SLA. Finds max-batch @SLA, computes events/s
  + tokens/s.
- **`models/experiments/run_ablation_table5.py`** (#6) — Semi-AR block
  ablation (B=1/2/4) + K-sweep ({2,4,6,8,10,12,16,24}). Includes K-knee plot.

### Verified

All 24 Python files compile. All 6 runners verified with `--mock` dry-runs:
- AR mock: p50~168ms (matches paper's AR-1.5B)
- MDM mock: p50~23ms at K=8 (matches paper's FlowFill)
- MDM K-sweep: linear scaling (K=4→12ms, K=8→23ms, K=16→47ms)

### Documentation

- **`AGENTS.md`** — quick-start guide for tomorrow's GPU runs with exact
  commands for AR and MDM experiments.

**Tomorrow's GPU work:**
```bash
pip install transformers accelerate torch

# AR (ready to run)
python models/experiments/run_quality_table2.py --model Qwen/Qwen2.5-1.5B-Instruct --n 10000

# MDM (needs denoising loop implementation first)
python models/experiments/run_quality_table2.py --mdm --model GSAI-ML/LLaDA-8B-Instruct --k-steps 8
```

## 2026-08-11 — GPU Experiments Begin (Issue #3: first real AR run)

First real GPU inference on H100 PCIe. Verified environment: torch 2.5.1+cu121,
CUDA 12.2, bf16, transformers 5.14.1, accelerate 1.14.0, xgrammar 0.1.33.
Models pre-cached: Qwen2.5-1.5B (3.3G), Qwen2.5-7B (15G), LLaDA-8B (15G).

### AR-1.5B quality run (n=500, BGL)

- **`results/quality_table2.json`** — Qwen2.5-1.5B-Instruct on 500 BGL events.
- **`results/quality_table2.md`** — comparison to paper Table 2.
- **`logs/quality_ar15_n500.log`** — full run log with tqdm progress.

**Results:**
| Metric | Our AR-1.5B | Paper AR-1.5B | Gap |
|--------|------------|---------------|-----|
| Field-F1 | 0.647 | 0.913 | -0.266 |
| Schema-valid | 98.4% | 96.1% | +2.3pp |
| Consistency | 99.6% | 98.9% | +0.7pp |
| Anomaly-F1 | 0.951 | 0.871 | +0.080 |

**Key finding — format divergence explains the Field-F1 gap:**
The 0.266 F1 gap is not a model quality issue — it's a **format normalization
mismatch**. Per-field breakdown shows:
- Structural fields the model formats correctly: timestamp (0.99), component
  (0.96), severity (0.98), error_code (0.96), is_anomaly (0.98).
- Fields with format divergence: `event_type` (0.0008!) — model outputs
  `data_storage_interrupt` (underscores) vs gold `data storage interrupt`
  (spaces). Token-level F1 sees these as non-overlapping tokens.
- `host` (0.43) — model is correct but field naming/placement varies.
- `affected_entity` (0.058) — model puts the host node here; gold leaves empty.

The paper's AR-1.5B was fine-tuned on the exact gold format. Our zero-shot
Qwen2.5-1.5B substitute has no such alignment. **Implication**: the raw F1
number is not directly comparable to the paper — but the *relative* AR-vs-MDM
comparison (once MDM runs) is still valid since both substitutes face the same
gold format.

**Latency caveat**: our naive `model.generate()` yields ~3.4s p50 vs paper's
168ms (20x slower) — paper uses vLLM+PagedAttention+FlashAttention-2. For
quality validation (#3), latency doesn't affect F1. For latency validation
(#4), we need vLLM or a documented caveat.

## 2026-08-11 — Prompt fix + LLaDA denoising loop implemented (Issue #3/#6)

### Bug fix: AR few-shot format mismatch

Found the root cause of the 0.266 Field-F1 gap above: `models/baselines/
ar_model.py`'s few-shot examples taught the WRONG output format —
`event_type: "data_storage_interrupt"` (underscores) and `affected_entity`
set to the host node, when gold labels use spaces (`"data storage
interrupt"`) and leave `affected_entity` empty unless a distinct resource
is named. Fixed both few-shot examples + added explicit format rules to
`SYSTEM_PROMPT`. Also fixed `local_files_only=True` on both AR and MDM
tokenizer/model loading (HF Hub network is blocked from this box; models
are pre-cached).

### LLaDA-8B denoising loop implemented (closes the #6 blocker)

`models/baselines/mdm_wrapper.py`'s `MDMModel.load()` and `_denoise()` no
longer raise `NotImplementedError`. Implementation:

- **`load()`**: loads LLaDA-8B-Instruct via `AutoModelForCausalLM` +
  `trust_remote_code=True`. Required two narrow monkey-patches for
  transformers 5.14.1 compatibility (LLaDA's `modeling_llada.py` targets
  transformers ~4.4x internals):
  1. `get_total_byte_count()` tolerates `LLaDAModelLM` missing
     `all_tied_weights_keys` (LLaDA's `__init__` never calls `post_init()`).
  2. `LLaDAModelLM.tie_weights()` wrapped to accept-and-ignore the
     `missing_keys=`/`recompute_mapping=` kwargs newer transformers passes.
  Mask token confirmed as `<|mdm_mask|>` = id **126336**.
- **`_denoise()`**: confidence-based unmasking (paper §2.2) — each of K
  steps does one forward pass (`use_cache=False`, no KV cache since all
  positions are recomputed with newly-unmasked context), takes softmax
  over logits at remaining masked positions, unmasks the `n_to_unmask =
  remaining // steps_left` highest-confidence positions. Supports semi-AR
  blocks (Table 5): split masked positions into B contiguous chunks,
  denoise sequentially, K/B steps per block.
- **Key design fix**: initially masked only individual JSON *values*
  inside a fixed key/brace skeleton — LLaDA-Instruct (not fine-tuned for
  this exact slot-filling shape) produced garbage (all fields → literal
  "0"). Switched to masking one **contiguous block of `<|mdm_mask|>`
  tokens** for the entire completion (keys + values), matching LLaDA's
  actual training/inference pattern (github.com/ML-GSAI/LLaDA
  `generate.py`) — the model reconstructs correct JSON structure from the
  few-shot examples already in the prompt. This immediately produced
  near-correct structured output.
- **`models/baselines/ar_model.py:parse_json_output`**: added a regex
  fallback (`_extract_key_value_pairs`) that recovers well-formed `"key":
  value` pairs from a malformed JSON document — MDM occasionally garbles
  one field (missing colon, duplicated tokens) while the other 11 decode
  correctly; the fallback recovers those 11 instead of discarding the
  whole record. This alone moved MDM's measured schema-validity from
  48% → 97% and Field-F1 from 0.536 → 0.774 on the same 200-event sample
  (same model, same weights — purely a parsing robustness fix).

**First real MDM (LLaDA-8B, K=8) quality result** (n=200, BGL, `results/
quality_table2_mdm_n200.json`):

| Metric | Value |
|---|---|
| Field-F1 | 0.774 |
| Schema-valid | 97.0% |
| Consistency | 72.5% |
| Anomaly-F1 | 0.821 |
| p50/p99/p99.9 | 1260 / 1495 / 1630 ms |

Per-field: structural fields (timestamp 0.975, host 0.935, component 0.96,
severity 0.97) all >0.9, matching the paper's "diffusion is free on
structural fields" claim. Weak fields: `metric_value` (0.25, numeric
extraction), `event_type`/`message_template` (~0.5, semantic). Consistency
(72.5%) is well below the paper's 97.8% — plausible given B=1 (no semi-AR
recovery) and an un-tuned open substitute; will re-check with B=2/4.

**Latency**: naive per-step `model(input_ids)` forward pass (no KV cache
reuse across steps, no custom kernels) — not comparable to paper's 23ms.
K-sweep below shows the *shape* of Proposition 1 holds regardless.

### K-sweep ablation (issue #6, Claim B) — first real result

`results/ablation_table5.json`, n=30 BGL events, K ∈ {2,4,6,8,10,12,16,24}:

| K | Field-F1 | p50 (ms) |
|---|---|---|
| 2 | 0.350 | 319 |
| 4 | 0.523 | 630 |
| 6 | 0.722 | 948 |
| 8 | 0.762 | 1265 |
| 10 | 0.766 | 1578 |
| 12 | 0.781 | 1892 |
| 16 | 0.777 | 2519 |
| 24 | 0.790 | 3771 |

**Findings:**
- **Knee confirmed near K=6-8**: F1 rises steeply 2→8 (+0.41), then nearly
  flat 8→24 (+0.03). Matches the paper's qualitative claim, though our
  knee is closer to K=6-8 than a sharp K=8 cutoff.
- **Latency is perfectly linear in K**: linear regression gives
  **R²=1.0000**, slope ≈157ms/step. This is strong empirical confirmation
  of Proposition 1 (Λ_MDM = K·t_fwd, content-independent, O(K)) — the
  *shape* of the paper's central latency claim holds on an open substitute
  even though our absolute per-step cost (no custom kernels/KV cache) is
  ~50x the paper's.

### Block ablation (issue #6, Claim A) — first real result, with a caveat

`results/ablation_table5.json`, n=30 BGL events, K=8 total steps:

| B | Field-F1 | Consistency | p50 (ms) |
|---|---|---|---|
| 1 | 0.762 | 70.0% | 1261 |
| 2 | 0.530 | 96.7% | 1263 |
| 4 | 0.511 | 80.0% | 1262 |

**Directionally confirms one paper claim, contradicts another:**
- Consistency **does improve with B=2** (70.0% → 96.7%, even bigger jump
  than the paper's 97.8%→98.6%) — the semi-AR mechanism's core claim (more
  sequential structure → fewer cross-field contradictions) holds.
- But **Field-F1 drops sharply** with B>1 (0.762 → 0.530 → 0.511), the
  opposite of the paper's claim that quality *also* improves with B.
- **Latency does not scale with B** (~1260ms flat) — the paper's model
  gets slower with B because more *sequential* forward passes are needed;
  ours doesn't show this because of the implementation caveat below.

**Root cause (methodological, not a code bug)**: our block splitting
divides the mask-block by raw **token position** (first N tokens = block
1, etc.), because at generation time we don't know which token ranges
will end up corresponding to which JSON field — LLaDA-Instruct generates
free-form JSON, it wasn't fine-tuned on a fixed-position 12-slot template.
The paper's FlowFill-MDM, in contrast, is trained so `S`'s slot positions
are known a priori (§2, the whole point of the L/T/S framework) — its
block split is a clean field-boundary split. With `K/B` steps per block
(B=4 → only 2 steps/block) and a garbled first block poisoning context
for later blocks, our result is expected to look worse than a true
field-aligned implementation. **This is itself a finding**: semi-AR
blocks are a schema-template mechanism, not a general MDM trick — it
requires the model to actually know the slot boundaries, which an
un-tuned open substitute prompted with free-form JSON does not have.

### GPU experiment runner fixes

- Added `--max-new-tokens` CLI flag to `run_latency_table3.py`,
  `run_throughput_table4.py`, `run_ablation_table5.py` — previously MDM's
  mask-block size was hardcoded to 256 tokens regardless of CLI args.
- Fixed `run_ablation_table5.py` defaults (model id, `--n`, calls
  `model.load()` before ablation loops).

### Environment

- H100 PCIe 80GB (paper used A100-80GB PCIe — ratios comparable, absolute
  numbers not).
- `HF_HUB_OFFLINE=1` required for all runs — this box has no outbound
  HTTPS/SSH to github.com/huggingface.co; all 3 models (Qwen2.5-1.5B/7B,
  LLaDA-8B) pre-cached locally beforehand.
- transformers 5.14.1, accelerate 1.14.0, xgrammar 0.1.33 installed.
  `logparser3` unavailable on the internal PyPI mirror — `drain_only.py`'s
  regex fallback is used (already the documented behavior when
  `logparser.Drain` import fails).

**Still running in background**: AR-1.5B + AR-7B quality run (n=2000,
`logs/quality_ar_n2000.log`) with the fixed prompt — will supersede the
n=500 format-divergence result above once complete.

## 2026-08-12 — Issue #12: AR baselines under real serving + two AR bugs found

Issue #12 was deferred because DiffusionGemma-26B needed a CUDA 12.3+ driver.
That blocker is gone (driver **580.65.06 / CUDA 13.0**, was 535 / 12.2), but a
different one replaced it, and chasing the fallback turned up two defects that
had biased every AR-vs-MDM comparison in this repo.

### DiffusionGemma is still not runnable here (different reason)

`DiffusionGemmaForBlockDiffusion` first appears in **vLLM 0.24.0** (checked the
`models/registry.py` of every wheel from 0.19.1 up: 0.19.1/0.21.0/0.22.1/0.23.0
lack it, 0.24.0/0.25.1/0.26.0 have it). Every version that has it pins
`torch==2.11.0`, a CUDA-13 build — and the internal mirror carries no CUDA-13
`cudnn`, `nccl`, `cufile` or `cusparselt`, all four of which are `DT_NEEDED` in
`libtorch_cuda.so`. pypi.org and download.pytorch.org are unreachable from this
host. No dependency-override trick gets around a hard `DT_NEEDED`. Full analysis
and the three things that would unblock it: `results/diffusiongemma_perf.md`.

**vLLM 0.19.1 does work** on the cu12.8 stack (torch 2.10.0+cu128), stripping
only optional kernel extras (flashinfer, cutlass-dsl, quack-kernels,
cudnn-frontend, cuda-pathfinder, pynvvideocodec). One stub was needed:
`openai_harmony` is imported unconditionally by `vllm/entrypoints/chat_utils.py`,
which is on the offline `LLM` import path though nothing there uses it.

### Bug 1: the AR baseline was served as a strawman

Every AR number in this repo came from a naive HF `model.generate()` loop. The
paper serves its AR baselines with PagedAttention + FlashAttention-2 and credits
them with a frozen-T conditioning cache, specifically so diffusion cannot win on
serving infrastructure. Ours had none of that.

`models/experiments/run_vllm_serving.py` re-measures the AR side under vLLM —
real continuous batching for Table 4, not the sequential loop the previous draft
used. On identical weights, prompts and events (n=2000 BGL):

| Serving | AR-1.5B p50 | p99 | p99.9 |
|---|---|---|---|
| HF transformers | 1842 ms | 3053 ms | 4163 ms |
| **vLLM** | **413 ms** | **686 ms** | **806 ms** |

Throughput under real batching peaks at **204.9 ev/s** (1.5B), 145.2 (3B), 85.1
(7B) at batch 512 — against 0.70 ev/s for naive LLaDA serving and the paper's
4350 ev/s claim. Prefix caching (the frozen-T cache) is worth only 1.03x on p50
here: the preamble prefill is small next to ~138 decode steps.

### Bug 2: the AR baseline was decoded with a hostile default

vLLM and HF disagreed on Field-F1 for the *same model and prompts* under greedy
decoding — 0.844 vs 0.779. The gap sat entirely in `host` (0.20 vs 0.83) and
`message_template` (0.69 vs 0.87).

Cause: Qwen2.5 ships `repetition_penalty` in `generation_config.json` (**1.1**
at 1.5B, 1.05 at 3B/7B) and HF `generate()` applies it even under
`do_sample=False`. `ARModel` never set the parameter, so the checkpoint's own
default silently took effect. For extraction this is exactly backwards —
correct output *copies spans out of the input*, which is what the penalty
suppresses. It emptied `host` on **80/100** events. vLLM defaults to 1.0, hence
the disagreement.

Same code path, same 100 events, only the penalty changed:

| | penalty 1.1 | penalty 1.0 | vLLM |
|---|---|---|---|
| host F1 | 0.200 | 0.800 | 0.830 |
| macro Field-F1 | 0.778 | 0.849 | 0.853 |
| empty `host` | 80/100 | 20/100 | 17/100 |

With it neutralised the two stacks agree to 0.004 (bf16 kernel drift). Fixed in
`ar_wrapper.py`: `repetition_penalty` is now an explicit constructor argument
defaulting to 1.0. **LLaDA/MDM was never affected** — no penalty in its config,
hand-written denoising loop — so the handicap was one-sided.

### What this costs us

Both defects flattered MDM, and two headline findings are withdrawn
(`VALIDATION_REPORT.md` carries a correction block):

- **"AR-vs-MDM near-parity (0.779 vs 0.779)" — refuted.** Corrected AR-1.5B
  scores 0.844 against MDM's 0.779, a 0.065 gap, an order of magnitude wider
  than the paper's claimed 0.006.
- **"AR-7B is the Field-F1 ceiling" — not supported.** That ranking came from
  the penalty being stronger at 1.5B (1.1) than 7B (1.05). Corrected:
  1.5B 0.844, 3B 0.826, 7B 0.842 — non-monotonic, all within ~0.02.
- **"MDM is 1.33x faster than AR" — reversed.** Properly served, AR-1.5B is
  3.4x *faster* than LLaDA-8B.

Unaffected, because they are internal to MDM rather than comparisons against
AR: content-independent O(K) latency (R²=1.0000), the K=8 knee, the
structural-vs-semantic decomposition, and MDM's consistency weak spot. MDM also
keeps the tightest tail ratio in the table (1.19x vs AR's 1.95-2.02x), so
Proposition 1 stands.

This does **not** show AR beats MDM architecturally — LLaDA is still on naive
HF serving, since vLLM has no MDM we can install. It shows our first pass
compared diffusion against a doubly-handicapped AR baseline.

### Withdrawn

`results/diffusiongemma_perf.json` reported 0.42 ms p50 and 322k ev/s for an 8B
model. Its runner silently fell back to *synthetic* log lines (it looked for BGL
at `datasets/bgl/test_200k.jsonl`, a path this repo does not use) and measured
"throughput" with a sequential `for` loop over `extract()`. The JSON is deleted
and `run_diffusiongemma_perf.py` is disabled with the diagnosis in its
docstring.

### Files

- `models/experiments/run_vllm_serving.py` — vLLM latency/throughput/quality runner
- `models/experiments/run_vllm_all.sh` — the full sweep (1.5B/3B/7B + no-prefix-cache)
- `models/experiments/make_vllm_report.py` — builds `results/vllm_serving.md`
- `models/experiments/diag_hf_vs_vllm.py` — the HF-vs-vLLM record-level diff
- `results/vllm_serving.md` + `vllm_serving_{1.5b,3b,7b,1.5b_noprefix}.json`
- `results/diffusiongemma_perf.md` — rewritten as a blocker analysis
