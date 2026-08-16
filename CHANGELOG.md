# Changelog

All notable changes to this project. Dates are 2026-08-15 unless noted.

## 2026-08-16 — Issue #112 resolved: AR-30B re-measured, headline corrected

The AR-30B arm was re-measured on the pinned stack (vLLM 0.25.1, torch
2.11.0+cu130, A100 80GB) with the corrected harness. All four artifacts
now carry provenance blocks; `verify_numbers.py` exits 0. The paper
compiles with 0 errors, 0 undefined refs, 21 pages.

### What changed (old → new)

| Quantity | Old (gold-leaked) | New (pinned stack) |
|---|---|---|
| arBase | 79.1 ms | 34.6 ms |
| arSlope | 47.7 ms/token | 7.2 ms/token |
| crossoverL | 6 | 48 |
| F1_9 (AR-30B) | 1.0 (impossible) | 0.870 |
| Headline ratio | 15.6× | 2.5× |
| Batch multiplier | 77× | 24.7× |
| Serial/batch slope ratio | 73× | 23× |
| Energy (AR-30B) | 10.1 J (batch) | 118.7 J (serial) |
| Energy ratio AR/DG | 2.4× (batch/serial) | 28.3× (serial/serial) |
| Cost/event | $0.0076 | $0.0012 |
| Annual cost gap | $26,000 | $2,600 |
| Per-million cost | $7,100 | $1,200 |

### What was fixed

- **Harnesses** (`run_issue111_{qwen3,latency,batch}.py`, `run_energy.py`): raw
  input (`bgl.raw`), gold paired by index, no `enforce_eager`, serial per-event
  timing, `_dict_to_record`, `capture_provenance`, `enable_thinking=False`.
- **Recurrence guard** in `build_chat_messages`: rejects gold JSON records.
- **Verifier** (`verify_numbers.py`): macro→artifact value check, provenance
  required for cited artifacts.
- **Appendix I**: correction 9 added (8→9 priced corrections).
- **Paper**: all AR-30B numbers propagated to macros, tab:niche, tab:cost,
  tab:contrast, tab:summary, abstract, §1, §3, §4, §5, §7, §8, Appendix 11/16/17.
  Threats paragraph rewritten from "unverified" to "corrected and re-measured."

### What survived

The qualitative contribution: DG's staircase cost shape wins at typical
output lengths (crossover at L=48, natural output at ~126 tokens). The
margin is narrower (2.5× not 15.6×) but the architecture is structural.

---

## 2026-08-16 — Issue #112 Phase 1: AR-30B harness fixed, verifier hole closed (no GPU)

The AR-30B arm was measured on a harness that (a) fed the gold record file to
the model as input, (b) disabled CUDA graphs (`enforce_eager`), (c) ran vLLM
0.11.0 vs the pinned 0.25.1, and (d) scored a wrong 9-field subset via a broken
`Record(**kwargs)` constructor that silently emptied every prediction — so
`field_f1_9` was `1.0` in every cell. Phase 1 (code) is done; the numbers are
**not** changed and now fail the build until GPU re-measurement.

- **Harnesses** (`run_issue111_{qwen3,latency,batch}.py`): raw input
  (`bgl.raw`) + gold (`bgl.jsonl`) paired by index; `enforce_eager` dropped,
  `enable_prefix_caching` on; serial per-event timing (was batched throughput
  labelled "serial"); `_dict_to_record` + 9-field macro from `field_f1`;
  `capture_provenance` block on every artifact. Latency harness now writes the
  cited `issue111_qwen3_serial_sweep.json` (was an orphan); gold-leaked
  `issue111_qwen3_latency_sweep.json` removed.
- **Recurrence guard**: `build_chat_messages` raises if its input parses as a
  gold JSON record — catches the #112 defect class for every harness.
- **Verifier** (`paper/verify_numbers.py`): docstring/exit-code made honest;
  macro→artifact value check for `arBase`/`arSlope`/`crossoverL`; missing
  provenance is now an ERROR (not a warning) for the four cited AR-30B
  artifacts. Exits 1 with 6 errors until re-measured.
- **Paper**: `07_threats.tex` paragraph updated to past tense (harness fixed,
  figures inherited, re-measurement pending); "unverified over as a bound" typo
  → "unverified upper bound". No numbers changed.
- **Remaining (GPU)**: re-run the three harnesses + `run_energy.py` (serial) on
  vLLM 0.25.1; refit cost law; propagate to macros/tables/prose; Appendix I
  correction 9.

---

## 2026-08-16 — Evidence gaps filled, application claims strengthened

### Issue #69: AR-1.5B measured at L=512/1024, energy, K=4 p99
- **AR-1.5B L=512: 2005 ms** (was extrapolated 826 ms — 2.4× too low)
- **AR-1.5B L=1024: 4026 ms** (was extrapolated 1652 ms — 2.4× too low)
- **AR-1.5B energy: 5.1 J/event** (net, was estimated ~5.6 J — close)
- **DM K=4 p99: 394.3 ms** (was estimated ~380 ms — meets 400 ms SLA)
- AR-0.5B: not available offline, removed from comparison
- **Impact:** all EDR-021/022 application claims STRENGTHENED. DG advantage at L=512 is 4.0× (not 1.65×)
- Root cause of extrapolation error: AR-1.5B rambles when forced past natural output length (~123 tokens), vLLM processes all forced tokens

---

## 2026-08-15 — Paper draft + pinned harness + cost laws

### Infrastructure (#57)
- **Pinned stack:** vLLM 0.25.1, torch 2.11.0+cu130, A100 80GB PCIe
- **Provenance helper:** `models/baselines/provenance.py` — shared, records model actually loaded
- **9-field scoring:** `metrics/quality_metrics.py` — drops 3 degenerate semantic fields (event_type, affected_entity, remediation_hint)
- **Parser fix:** `models/baselines/ar_model.py` — first-object-wins (was last-wins via `rfind`). AR-3B L=512 F1: 0.599→0.826
- **Gold alignment:** `models/baselines/gold_alignment.py` — assertion + shuffled-gold canary
- **Artifact audit:** 98 files scanned, 4 model/model_type mismatches corrected with `.provenance.json` sidecars
- **Anchor re-measurement:** AR-3B CV=0.05%, DG CV=0.25% — both <2% threshold
- **Discrepancy explanations:** AR 1617→1915ms = vLLM 0.25.1 vs 0.7.2; DG 418→379ms = K=32 vs K=8

### Cost laws (#44, #68)
- **Two-variable law:** T_DG(L_in, L_out) — DG input slope 0.123 ms/prompt-tok, 2.6× steeper than AR (0.047)
- **Staircase confirmed:** sharp edges at 256-token canvas boundaries, flat between, step=119.6ms median
- **Base decomposition:** 93.5% prefill+fixed, 6.5% denoising at K=8. F1 collapses below K=4, flat above K=8
- **HDFS generalization:** step=120.5ms on HDFS (BGL: 118.5ms). AR HDFS p50 matches BGL within 0.03%

### Decision procedure (#64)
- `models/decision/procedure.py` — 5 guards (truncation, throughput, length, quality, default-DG)
- `models/decision/cli.py` — CLI (`python -m models.decision`)
- `models/decision/test_procedure.py` — 11/11 tests pass
- Crossovers: L*=59 tokens (K=8), λ*=2.6 ev/s — reproduced from law constants

### Checklist (#65)
- `CHECKLIST.md` — 5 harness defects, 4 of 5 biased against AR
- `paper/verify_numbers.py` — asserts all 5 magnitudes, 0 errors

### Canvas tuning (#66)
- `canvas_length=256` is tunable in DG config
- canvas=128: free improvement (base -2.4%, F1 +0.011)
- canvas=64: not viable (base +48%, F1 -0.027)

### Batching (#42)
- **Retitle:** batching cap stated as correlation, not mechanism
- DG multiplier 1.18× vs AR 77–86× — candidates: bidirectional attention, MoE, kernel maturity

### Paper draft
- `paper/main.tex` — 15 pages, 9 sections, 7 figures, compiles cleanly
- `paper/macros.tex` — all numbers shared between paper and CHECKLIST.md
- `paper/BACKBONE.md` — 16 green, 2 amber, 0 red claims
- `paper/CHARTS.md`, `paper/NUMBERS.md` §0

### EDRs (Experimental Decision Records)
- `docs/edr/` — 16 records, all 19 claims covered, all 9 sections backed
- Format: Hypothesis → Design → Prediction → Result → Decision → Impact

### Scripts added
- `models/experiments/run_anchor_remeasure.py` — 5×2 anchor cells
- `models/experiments/analyze_input_sweep.py` — #44 input-length analysis
- `models/experiments/analyze_ksweep.py` — #44/#68 K-sweep + canvas analysis
- `models/experiments/analyze_generalization.py` — #67 HDFS vs BGL
- `models/experiments/audit_artifacts.py` — #57 artifact audit
- `models/experiments/run_energy_measurement.py` — #5 energy (queued)
- `models/experiments/run_memory_probe.py` — #6 memory (queued)
- `models/experiments/run_infilling.py` — #7 infilling (queued)
- `paper/generate_figures.py` — 7 figures from committed artifacts

### Issues closed
- #57 — pinned harness
- #44 — two-variable cost law
- #42 — retitle batching mechanism
- #65 — checklist
- #66 — canvas tuning
- #68 — canvas step edges

### Issues in progress
- #67 — generalization (LLaDA dInfer running on GPU)
- #58 — deployment envelope (P1/P2 arithmetic done, P5 queued)
- #59 — infilling (script written, queued after #67)

---

## 2026-08-14 — Pre-correction experiments

### GPU experiments (#45, #47, #49, #50)
- AR-3B + DG on A100 80GB, vLLM 0.7.2
- Per-length predictions saved (#43)
- Input-length sweep (#44 prep)
- Batching test (#42 prep)
- HDFS AR-3B (#50, AR arm only)

### Analysis
- Issue #43: per-length predictions revealed parser bug (last-wins)
- Issue #38: completeness metric fix (0%→100%)
- Issue #48: M/D/1 not predictive under bursty arrivals
- Issue #39: copy-weakness analysis

---

## 2026-08-11–13 — Early experiments

### Baselines (#3–#8)
- FlowFill reproduction: quality (Table 2), latency (Table 3), throughput (Table 4), ablation (Table 5)
- HF transformers stack, no vLLM
- 12-field macro F1 (now known to be contaminated)

### Serving (#12)
- vLLM serving for AR models
- Decoding config fix: repetition_penalty 1.1→1.0 (F1 0.779→0.844)

### Cost laws (#14–#21)
- Rate-vs-lag curves, output-length experiments
- Campaign B (vLLM 0.25.1) — the pinned stack
- First cost-law fits: AR linear, DG staircase

### Negative results (#24–#27)
- #24: iterative refinement — DM improves but so does AR
- #25: bounded deadline — AR completes (metric was buggy, fixed in #38)
- #27: perturbation — DM worse than AR in 10/10 categories
