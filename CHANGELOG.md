# Changelog

All notable changes to this project. Dates are 2026-08-15 unless noted.

## 2026-08-17 — Issues #119–#120 resolved: cascade measured, tail latency at n=2000

### Issue #120: Tail latency at n=2000 (blocker, closed)

Re-ran DG and AR-3B anchors at n=2000 (was n=300 where p99.9==max, statistically
meaningless). Results:

| Metric | Old (n=300) | New (n=2000) |
|---|---|---|
| DG p99.9 | == max (1211ms) | 557ms (distinct from max=1166ms) |
| DG max/median | 3.21× | 3.07× |
| DG CV | 20.9% | 16.9% |
| AR-3B CV | 0.2% | 0.1% (unchanged) |

Bootstrap 95% CI for DG p99.9: [493, 1166]ms — still wide, but the point
estimate is now distinct from max and stable across sample sizes. Updated
macros (`dgEventCV`, `dgMaxMed`, `arEventCV`) and threats section.

### Issue #119: Anytime K-cascade (closed)

Measured the K=2→K=8 cascade on BGL (n=200, canvas=256) using issue114
per-record predictions. The cascade uses output completeness (all structural
fields filled) as the escalation signal — vLLM's diffusion logprobs return
certainty for committed tokens, so per-token confidence is uninformative.

Results:
- 29.5% of records resolved at K=2; 70.5% escalated to K=8
- Cascade F1=0.848 (drop of 0.005 vs fixed K=8=0.853)
- Cascade latency=465ms (slower than fixed K=8=376ms — the K=2 overhead is
  not amortised at 70% escalation rate)

The structural property (input-dependent latency) is confirmed — DG can
check output quality mid-process and decide whether to invest more compute.
AR cannot. The practical latency benefit is task-dependent: it helps on
tasks where most inputs are easy, not on BGL where 70% need K=8.

Updated §5 (survives) with measured cascade results. Updated §7 (threats):
replaced "not yet measured" with findings. Added `run_anytime_cascade.py`.

---

## 2026-08-16 — Issues #112–#118 resolved: AR-30B re-measured, paper corrected, HumanEval generalization

### Issue #112: AR-30B gold-leak harness (blocker, closed)

The AR-30B arm was measured on a harness that (a) fed the gold record file to
the model as input (prompting with the answer), (b) disabled CUDA graphs
(`enforce_eager`), (c) ran vLLM 0.11.0 vs pinned 0.25.1, and (d) scored a wrong
9-field subset via a broken `Record(**kwargs)` constructor that silently emptied
every prediction (F1=1.0 from empty-vs-empty scoring). Re-measured on the pinned
stack (vLLM 0.25.1, torch 2.11.0+cu130, A100 80GB, n=300, serial).

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

Harness fixes: raw input, no eager, serial timing, `_dict_to_record`,
`capture_provenance`, `enable_thinking=False`. Gold-leak guard in
`build_chat_messages`. Macro→artifact value check in `verify_numbers.py`.
Correction 9 added to Appendix I (8→9 priced corrections). Grill pass 8
fixed readability, stale crossover values, removed resolved threats.

### Issue #113: Pin stack, harness refactor (blocker, closed)

Pinned exact versions in `pyproject.toml` (vllm==0.25.1, torch==2.11.0,
transformers==5.15.0). Added `.python-version`, `repro_env.txt`,
`requirements-frozen.txt`. Fixed `run_issue72_grammar.py` (same
`FIELD_NAMES[:9]` + `Record(**kwargs)` bug class). Deleted duplicate
`NINE_FIELD_NAMES` from `provenance.py`. Updated `procedure.py` constants
(crossover 65→48, slope 6.29→7.2, batch 80→24.7). Updated AGENTS.md.
11/11 tests pass.

### Issue #114: Joint canvas×K sweep (blocker, closed)

20-cell grid (canvas ∈ {64,128,256,512} × K ∈ {2,4,8,16,32}, n=200/cell).
Key finding: the paper's recommendation (canvas=256, K=8: p50=376ms,
F1=0.853) is **Pareto-dominated** by canvas=128, K=8 (p50=368ms, F1=0.864).
K knee shifts left at smaller canvas. Canvas=64 not viable even at K=32.

### Issue #115: HumanEval cost-shape generalization (blocker, closed)

Built `run_humaneval.py` with execution sandbox (164 problems, serial).
Cost-shape law holds on code: AR-30B = 6.4 ms/tok (vs 7.2 on logs), DG is
flat staircase (280ms). AR/DG ratio = 2.9× (vs 2.5× on logs). K knob works
(K=2→0.146, K=4→0.232, K=8→0.287).

### Issue #116: Deadline-bounded pass@1 (closed)

AR collapses under deadline (0.72→0.09 at D=200ms, code_validity 94%→23%).
DG stays flat (0.287 at all deadlines). Crossover at D≤400ms: DG wins
despite being worse at code generation.

### Issue #117: Multi-turn code correction chain (closed)

Built `run_code_chain.py` (generate→execute→fix→re-execute, up to 5 rounds).
DG per-round latency stays flat (287→305ms, +6%), AR grows (511→620ms, +21%).
Both saturate by round 3. DG solves 95.7% cumulatively vs AR's 89.6%.

### Issue #118: Speculative decoding (closed, in b78ce83)

### Paper changes (all issues)

- New Appendix 18: HumanEval cost-shape generalization (baseline + deadline + chain)
- §2: generalization paragraph referencing code generation
- §5: code chain evidence, anytime cascade measured results
- §7: tail latency n=2000 results, cascade measured (was "not yet measured")
- §8: conclusion updated (2.5× not 15.6×, $1,200 not $7,100)
- §9: correction 9 added (9 priced corrections, was 8)
- §13: joint canvas×K sweep subsection, stale crossover fixed
- Abstract: rewritten for readability
- Macros: arBase, arSlope, crossoverL, dgEventCV, dgMaxMed, arEventCV updated
- Paper: 23 pages, 0 errors, 0 undefined refs, verify_numbers.py exit 0

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
- `paper/CHARTS.md`, `docs/legacy/paper/NUMBERS.md` §0

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
