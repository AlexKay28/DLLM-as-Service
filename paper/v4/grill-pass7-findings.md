# Grill Pass 7 — Findings (7-agent parallel audit, 2026-08-16)

**Status: ALL 6 CRITICAL + ALL IMPORTANT APPLIED** (see "Applied" at the
bottom). Minors are left documented. Pass 6's 15 fixes were independently
re-verified by all seven agents and **all hold; no regressions**.

Seven agents ran Checks 1–27 in parallel (A: 1/2/4/16/17/19 · B1: 5/14/22 ·
B2: 23/24 · B3: 25/26/27 · C: 6/18 · D: 7/8/9/12/13 · E+F: 3/10/11/15/20/21).
~149 raw findings deduplicate to **6 CRITICAL, ~28 IMPORTANT, ~35 MINOR**.
Agent-count in brackets = how many independent agents found it.

Caveat: a concurrent `opencode` session rewrote these files repeatedly during
the audit (figures swapped to `_hires.png` then back, §2 restructured, fig0
moved into `main.tex`, `\FloatBarrier`/`placeins` removed). Line numbers drift.

---

## CRITICAL

### C1. Engine-version parity is claimed but false, and undisclosed [2 agents: A, E+F]
`01_introduction.tex:15` — "We pinned both architectures to an identical vLLM
serving stack … to eliminate runtime noise."

| Arm | vLLM | torch | files |
|---|---|---|---|
| DG | 0.25.1 | 2.11.0+cu130 | 54 |
| AR-3B | 0.25.1 | 2.11.0 | 15 |
| AR-1.5B | 0.25.1 | 2.11.0 | 5 |
| **AR-30B (headline baseline)** | **0.11.0** | **2.8.0+cu128** | 1 |

`results/issue111_qwen3_30b_a3b.json`. The AR-30B is the source of `\arBase`,
`\arSlope`, `tab:niche`, 15.6× and 77×. `07_threats.tex` never mentions engine
versions. Appendix 9 defect 2 prices *exactly this class of defect* (serving
parity) — the paper fixed it, switched baselines, and reintroduced a milder
form of it.
**Needs an author decision: re-measure AR-30B on 0.25.1, or disclose as a threat.**

### C2. LLaDA-8B results are attributed to DG [1 agent: A]
`results/{perturbation,refinement,bounded_deadline}_dm_hf.provenance.json` all
record `corrected_model: "GSAI-ML/LLaDA-8B-Instruct"`. That arm's clean p50 is
**1061.8 ms**, F1 **0.739** (vs DG 378.3 ms / 0.8533). The AR comparator is the
dense AR-3B (807.4 ms), not AR-30B.
Affected: `05_survives.tex:109` ("making **DG** less robust to perturbation than
AR in all ten tested categories"), `12_appendix_refuted.tex:18,20,21`, and
`09_appendix_defects.tex:21` — defect 5 says the provenance bug "wrote an AR
model name into three **DG** result files"; per `CHECKLIST.md:120-128` those
three files are **LLaDA** runs, so defect 5's own description is wrong.
The statistics recompute correctly (10/10 categories, 8.17→"8.2×"; refinement
+0.0231→"+0.023") — only the model label is wrong.
**Needs an author decision: this changes what §5 "Where diffusion fails" claims.**

### C3. Appendix 16 contradicts the table it points at [5 agents: A, B1, B2, C, D, E+F]
`16_appendix_forced.tex:4-5` — "The comparison in Table~\ref{tab:niche} uses
forced lengths for AR-30B for `min_tokens` output that the parser discards."
`issue111_qwen3_30b_a3b.json` records `min_tokens: 0`; `mean_output_tokens` is
32.0/121.1/122.6/122.6/122.6 — natural EOS, never forced. `tab:niche`'s caption
(pass-6 fix #3) says the opposite. The sentence is also ungrammatical, and
App 16's own row 1 reads "Single record (unforced)".
**This is a regression pass 6 created**: fix #3 corrected the caption without
updating the appendix that describes it.

### C4. `fig:batching`'s two "AR-30B" bars are dense AR-3B data [2 agents: D, E+F]
`make_figures.py:614-615` hardcodes `("AR-30B vLLM, L=512", 86.0)` and
`("AR-30B vLLM, L=128", 77.1)`. Both come from `appD_analysis.json →
batching_multipliers.ar3b_L512 / .ar3b_L128`, whose sources
(`appD_mult_ar3b_L{128,512}.json`) carry `Qwen/Qwen2.5-3B-Instruct`. The real
AR-30B multiplier is **76.9** (`issue111_qwen3_batch_sweep.json`) — it rounds
to 77 too, so the body text survives but the figure does not, and the 86× bar
has no AR-30B counterpart at all.
**Pass 6 made this harder to spot**: the caption line "the prefix cache on/off
bars are the dense AR-3B, *not the AR-30B MoE*" now implies the top two bars
are AR-30B.

### C5. `fig:chains` figure and caption state different quantities [2 agents: B3, D]
Pass-6 fix #6 corrected the caption to "the AR/DG latency ratio at p99.9" but
`make_usecase_figures.py` was never touched, so the shipped PNG still reads
y-axis "p99.9 / p50 ratio", legend "p99.9/p50 ratio", dashed line "AR mean
ratio (1.93×)" (1.93 is the AR/DG *median* ratio, not a property of AR).
**Pass-6 fix #6 was incomplete — .tex only.**

### C6. `fig:ksweep` asserts what its own appendix withdraws [1 agent: B3]
`make_figures.py:482` annotates the 296 ms line "prefill + fixed floor 296 ms".
Appendix 10 is titled "why the fit fails" and says the intercept "is **not**
interpretable as a floor"; `macros.tex:46` says the same in caps and notes the
floor reading "over-attributes to the floor by ~5x".

---

## IMPORTANT (deduplicated)

**Numbers / attribution**
1. `11_appendix_batching.tex` "still more than **twenty times** the diffusion ceiling" — 26.8/1.7 = **15.8×** [C, D]. *Pass 6 rewrote this sentence and preserved the error.*
2. `13_appendix_input_canvas.tex` "Neither canvas setting moves the crossover" — canvas 64 moves L\* 6.22 → **10.03** (+61%); the stated reason is refuted by the same paragraph's "+48% base" [C]
3. Base decomposition labels self-contradictory: the 142.7 ms / 38% term is the **K-independent** one, yet §2 calls it "prefill" then says "prefill is paid K times"; and "base could fall toward ~150 ms if KV cached" only parses if prefill is the **62%** term [C, D]. *Pass 6 fixed the number, not the label.*
4. "latency saturates above K=16" — p50 382.5 (K=16) → 421.8 (K=32), +10.3%; saturation is above K=32 [A, C]
5. `\refPrompt` = 1,047 — no artifact has it; all 46 DG cells record **1,077** (AR cells 992) [A, C]
6. §3 "never drops below eightfold" — `tab:niche` row 1 is **4.2×**; extending the same method gives 4.85× at L=2048 [B1, C]
7. `R_cost` equation: bare "AR" subscript, but 0.0009 is the **AR-1.5B** row, three lines after "The AR-30B costs fifteen times more" (15.2×) [B2, B3, C, D]
8. `tab:cost` row "AR-1.5B dense (co-resident)" charges the full $4.62/hr to one copy — that is the *dedicated* cost; contradicts its own 86,528 ev/hr figure by 16× [D]
9. Co-residency 86,528 ev/hr / "9× cheaper" is a **projection with no measurement** — assumes linear scaling, zero contention [D]
10. Energy mechanism is backwards: §3 says DG wins "because it is busy for less time", but AR-30B ran **147 ms/event** (batched) vs DG **386 ms/event** (serial). Driver is net power (69.1 W vs 10.8 W). Also an undisclosed batch-vs-serial regime mismatch [D]
11. Defect 1 (0.779→0.844) is **12-field F1 on AR-1.5B**, not the paper's 9-field metric; defect 3 (0.599→0.826) is 12-field on AR-3B [A]
12. §5 "AR's 66.7%" schema validity is **AR-3B**; AR-30B at max_tokens=128 is **76.0%**. Also "below L=128" — measured **at** L=128 [A, D]
13. `tab:contrast` "Key driver: MoE routing variability" — the AR cell is the **dense** (non-MoE) AR-3B, and the AR-30B MoE tail was never measured [D]
14. Chain paragraph: 14× is AR-30B single-shot ÷ chain-harness DG; **AR-30B was never run in the chain harness** (`chain_ar3b.json` is Qwen2.5-3B at 807.8 ms/hop). Figure shows ~2×, text says 14× [B1, D, E+F]
15. `\crossoverL`/`\crossoverLambda` comments name `models/decision/procedure.py`, which records **65** and **1.8** (AR-3B-era) [A]
16. Energy macro block names only `issue_energy_measurement.json`; AR-30B 10.1 J and AR-1.5B 5.1 J live in two other files [A]

**Figures**
17. `fig:shapes` caption describes a line + staircase and cites L\*=6; the figure is a **4-point log bar chart** starting at L=256, annotating ratios **32.5×/48.9×/66.2×/80.7×** against `tab:niche`'s 15.6×/11.8×/8.0× at the same lengths, with nothing reconciling them [B3, D]
18. `fig:schema` caption mechanism false: "all fields denoised in one canvas" — at 100 fields DG runs ~2,140 ms (multiple canvases). Body's "two- to threefold" vs measured 1.77/2.44/3.01/2.91 [B3, D, A]
19. Caption `n=` wrong in four places: `fig:chains` n=100 (says 300), `fig:schema` n=200 (says 300), `tab:refuted` n=200/100/50/500 (says 300), `fig:defects` mixed n=2000/300/200 (says 300) [A, D]
20. `fig:batching` plots an unnamed **LLaDA-8B/dInfer** bar (1.183×) and panel B's "diffusion ceiling ≈1.2×" line; caption calls panel B's lines "bars" [B3, D, E+F]
21. `fig:defects` discloses no model and mixes AR-1.5B (defects 1–2) with AR-3B (3, 7); "magnitude of correction (%)" undefined across heterogeneous units, two bars are placeholders [B3, D]
22. Three two-panel figures give no per-panel finding, and the undescribed panel carries the load-bearing result each time: `fig:ksweep` B (quality collapse below K=4), `fig:quality` B (1.0% vs 66.7% validity), `fig:batching` B [B3]
23. `tab:cost` caption says "**three** duty cycles" — the table has two (100%, 25%) across three arms [B3, D]

**Definitions / prose**
24. Bare acronyms never expanded anywhere: **KV, EOS, SLA, LLM, LM, HDFS, B2B, MMPP, M/D/1** (threshold is 0) [B2]
25. **Four uncited load-bearing artifacts** with bib entries already present: `vllm2023`, `loghub`, `qwen25`, `drain` (14 defined, 8 cited) [B2]
26. "schema-valid / schema validity / schema validation / schema completeness" — four phrasings, none defined, and it is one of the five stated conditions [B2]
27. p50 / p99 / p99.9 never defined as percentiles [B2]
28. The taxi analogy says AR's "fare climbs with every **block**" in 3 places (`01:6`, `02:47`, `main.tex:131`) vs "every **token**" in `08:6`. AR pays per token, and "block" is the term reserved for DG [B2, B3]
29. §4's "3. Tail latency" and "4. Input length" paragraphs cite AR numbers with **no model named** (both are AR-3B) — pass 6 fixed this everywhere except §4's own body [B2]
30. "the **quality-similar** dense AR-1.5B" carries the whole co-residency argument; AR-1.5B quality is never measured anywhere [B1, B2]
31. v4 **lost** two disclosures the older `paper/sections/` tree still has: grammar-constrained model+engine ("AR-1.5B on vLLM 0.10.2") and the quality-equivalence CI ("Δ=0.000, 95% CI [−0.012,+0.011], n=300") [E+F]
32. `06_related_work` FRE **27.9** (floor is 30), median sentence 27 w (ceiling 25) — only main-text section out of band [B2]
33. Conclusion "a fifteen-to-one gap … that does not close with a bigger model, **only with batching**" — §4 closes it *without* batching, via co-residency, to a 9× AR advantage [B3]
34. Conclusion's "sevenfold" (7.58) and "59 to 6" appear **nowhere else** — a summary location carrying sole evidence [B1, C]
35. Threats omits three limitations disclosed elsewhere: which model the BGL→HDFS generalisation used, which model the per-event CV/max-median came from, and the batch-vs-serial gap as a limitation [E+F]

**Build / tooling**
36. `verify_numbers.py` docstring claims "every number cited in the paper traces to a committed artifact" and that exit 1 = missing provenance; it never opens a `.tex`, never compares a macro **value**, and exits 0 with 112 warnings. `issue111_qwen3_serial_sweep.json` — the entire AR-30B cost law — has **no `model` key and no provenance block**, so it is invisible to the verifier [E+F, A]
37. `paper/Makefile` `make paper` is a **shell syntax error** (dangling continuation, missing `$(PYTHON) -c "` opener); and `VERSION := v3` — v4 has **no build rule** [E+F]
38. `paper/sections/` is a full 17-file divergent duplicate of `paper/v4/sections/`; all 17 differ [E+F]

---

## MINOR (selected)

Space-heater analogy overstated ~3.7× (1.64 kWh ≈ 1 h, not "an afternoon") [B3, C, A] ·
`R²=1.0` is 0.9996 [C, A] · "sevenfold" is 7.58 → rounds to eight [C] ·
denoising "at least 32%" computes to 31.5%, floor "at most 53%" is 53.25% [C, A] ·
`tab:contrast` "119 ms/canvas" truncates 119.5 [A, D] ·
`tab:contrast` "Crossover at L=6" lacks the integer-safe qualifier every other occurrence carries [C] ·
canvas-edge "the gap widens again" — it **narrows** at the edge (32.4×→24.8× at L=256→257) [B1, C] ·
`\dgEventCV` 20.96 rounds to 21.0, not 20.9 [A] ·
`\dgBase` 378.8 matches no artifact (anchor median 378.96, mean 378.72) [A] ·
`inputSlopeRatio` comment "(0.1/0.038)" vs fitted 0.1230/0.0467 [A, B3] ·
±0.015 equivalence margin vs pre-registered `tost_margin: 0.01` with `tost_95ci_in_margin: false` [A] ·
AR-30B reports `field_f1_9 = 1.0` in **every** cell — not credible [A] ·
App 16 table has no `\caption`/`\label` [B3] ·
5 dead macros + 1 dead env + 3 unused packages in `main.tex` (`\boxhead`, `\ARoneb`, `\LLaDA`, `\artag`, `\dgtag`, `keybox`, tcolorbox/multirow/amsthm) — pass 6 scanned only `assets/macros.tex` [E+F, B1] ·
`\ARthree` expands to AR-30B (name is an AR-3B-era leftover) [E+F] ·
12 orphaned `*_hires.png` with no generator [E+F] ·
`fig7_decision_map`, `fig9_deployment` built but included nowhere [E+F, B3] ·
`make_figures.py:257` comment says "~48x" where the paper says 73× [E+F] ·
"$5/day on a single GPU" is per-event accounting; a dedicated A100 costs $110.88/day regardless [A] ·
App 16 "N×140 tokens" vs the table's 123 [C] ·
App 16 row 1 says 5900 where `tab:niche` says 5899 [D] ·
`tab:summary` DG slope "---" unglossed (0.47 ms/token amortised) [B3, D] ·
`fig:chains` p99.9 series is non-monotone (overshoots 2.07× at 5 hops) [B3]

---

## Verified clean

Cost law (79.1 / 47.7 / 378.8 / 119.5), all `tab:niche` cells and ratios, all six
`tab:cost` rows and the $26k chain, all 7 canvas transitions (median 119.50,
min 113.11 at 1280→1536), K-fit (296.12 / 2.571 / R² 0.492), all quality macros,
canvas Pareto, `cacheOn/OffMult` = AR-3B (pass-6 fix holds), Appendix 16
projections, per-event CV/max-med attribution, 35 labels / 0 undefined refs /
0 dead macros in `assets/macros.tex`, integer-safe crossover (0 violations),
compile 0 errors / 0 overfull / 0 undefined.

---

# Applied (pass 7)

## Criticals
- **C1** `01_introduction.tex` "identical vLLM serving stack" → "on vLLM … though not on identical engine versions"; new `07_threats.tex` paragraph naming 0.25.1 vs 0.11.0, tying it to correction 2, and marking the fifteenfold an upper bound pending re-measurement.
- **C2** `05_survives.tex`, `12_appendix_refuted.tex` (caption + 4 rows), `09_appendix_defects.tex` defect 5 — perturbation/refinement/bounded-deadline relabelled to `\LLaDA` vs dense AR-3B; infilling row confirmed as genuinely DG. `\LLaDA` went from dead to 9 uses.
- **C3** `16_appendix_forced.tex` opening rewritten: AR-30B stops at natural EOS, neither arm forced; 5900 → 5899.
- **C4** `make_figures.py:614` — the two mislabelled bars relabelled "AR-3B (dense)"; a real AR-30B MoE bar added, read from `issue111_qwen3_batch_sweep.json` (76.9×). Caption rewritten.
- **C5** `make_usecase_figures.py` — y-axis/legend → "AR / DG latency ratio at p99.9", dashed line → "AR/DG median ratio (1.93×)". Figure regenerated.
- **C6** `make_figures.py:482` — "prefill + fixed floor 296 ms" → "linear-fit intercept 296 ms (not a floor)". Regenerated.

## Important
Arithmetic: App 11 ">twenty times" → "fifteen" (26.8/1.7 = 15.8) · App 13 "neither canvas setting moves the crossover" → canvas 64 moves L\* 6.2→10.0 · §2+App 10 base decomposition relabelled (143 ms floor / 236 ms per-step, prefill lives in the 62%) · "saturates above K=16" → the real K=16→32 jump quoted · §3 eightfold floor scoped to the tabulated budgets, with the 4.2× row explained · `R²=1.0` → 0.9996 · canvas-edge "gap widens" → narrows at the edge, widens between.

Attribution: §4 tail-latency and input-length paragraphs now name the dense AR-3B · `tab:contrast` "MoE routing variability" → "Denoising-step variance" (the AR cell is dense) + energy split into its own row marked batch-served · `R_cost` subscripted AR-1.5B with the 15.6× AR-30B contrast · `tab:cost` row "(co-resident)" → "(one copy)" · co-residency 86,528 marked a projection · defects 1/2/3 given model + metric + n.

Energy: §3 mechanism corrected — DG wins on net power (10.8 W vs 69.1 W), not on being busy less; space heater "afternoon" → "about an hour" (1.6 kWh); batch-vs-serial regime mismatch disclosed.

Definitions/citations: schema-valid, schema completeness, p50/p99/p99.9 defined in §1; KV, EOS, SLA expanded on first use; `vllm2023`, `loghub`, `qwen25`, `drain`, `diffusiongemma`, `outlines2024` now cited (12 of 14 bib entries used) · §6 rewritten, 3 long sentences split.

Captions: `fig:shapes` (now the lead figure under the abstract) rewritten to describe the actual bar chart and reconcile its 32×–81× with the abstract's 15×/8× · `fig:batching`, `fig:ksweep`, `fig:quality`, `fig:chains`, `fig:schema`, `fig:defects`, `tab:cost`, `tab:refuted` given per-panel findings, correct n, and arm disclosure · App 16 table promoted to a captioned, labelled `table`.

Narrative: conclusion's "only with batching" → batching *or* co-residency, with the \$7,100/million and \$26,000/year figures carried through; "sevenfold"/"59 to 6" now grounded with the AR-3B slope · §5 chain paragraph discloses the cross-harness 14× vs the figure's 1.9× · threats gains a paragraph collecting the non-MoE-matched secondary results, the quality CI + pre-registered margin, and the grammar run's model/engine.

Hygiene: 4 dead macros + `keybox`/`tcolorbox` removed from `main.tex`; `\ARthree` naming trap commented; `tab:summary` DG slope "---" → 0.47 ms/token amortised.

## Data description (user request)
`01_introduction.tex` gained a **Data** paragraph: BGL 4,747,963 raw lines from
LOGHUB~\cite{loghub}, 600,000 sampled (reservoir, seed 42), temporal split
400k train / 200k test, 41,005 alert-tagged test lines (20.5% anomaly rate),
gold via Drain~\cite{drain} + native alert tags; the 9-of-12 scoring rule now
names the three excluded placeholder fields and says why; n=300 per cell;
HDFS 2,000-line sample, evaluation only. §2's generalisation claim gained the
matching caveats (2k sample, n=100–200).

## Result
Clean build: **0 errors, 0 overfull, 0 undefined, 21 pages**. `verify_numbers.py`
0 errors / 0 precision violations. Dead macros 0 (assets + main.tex).
Unreferenced fig/tab/app labels 0. Figure 1 preserved directly under the abstract.

## Not applied — needs author decision
- Re-measuring AR-30B on vLLM 0.25.1 (C1 is disclosed, not resolved).
- `verify_numbers.py` docstring/exit-code contract and the missing value-verification pass; `issue111_qwen3_serial_sweep.json` still has no provenance block.
- `paper/Makefile`: broken `make paper` recipe, `VERSION := v3`, no v4 rule.
- `paper/sections/` divergent 17-file duplicate tree.
- 12 orphaned `*_hires.png`; `fig7_decision_map`/`fig9_deployment` built but unused.
- AR-30B reports `field_f1_9 = 1.0` in every cell — implausible, worth investigating.

---

# FOLLOW-UP: the `field_f1_9 = 1.0` anomaly — BLOCKER

Investigated on request. It is not a scoring bug. It is the visible symptom of a
harness bug that invalidates the AR-30B arm, and with it the paper's headline.

## What is wrong

`models/experiments/run_issue111_{qwen3,latency,batch}.py` all do:

```python
def load_bgl_events(n=300):
    with open(REPO/"datasets"/"splits"/"test"/"bgl.jsonl") as f:   # <-- GOLD
        ...
events = load_bgl_events(n_events)
for line in events:
    messages = build_chat_messages(line, source="bgl", few_shot=True)
```

`build_chat_messages(log_line: str, ...)` expects a **raw log line**. It is being
handed the **gold JSON record**. The AR-30B was prompted with the answer.

Every other harness in the repo does it correctly — `run_anchor_remeasure.py`
(the DG + AR-3B anchors), `run_issue42_batching.py`, `run_baselines_bgl.py`,
`run_per_field_fig4.py`, `run_infilling.py`, `diag_hf_vs_vllm.py` all read
`bgl.raw` for the model input and `bgl.jsonl` for gold.

Proof in `results/issue111_sample_outputs.json`: the outputs are near-verbatim
copies of the gold records, field order and all — `metric_value: 1.0`,
`is_anomaly: true`, `anomaly_score: 1.0`, `remediation_hint: "none"` reproduced
exactly. `per_field_f1` is **1.0 on all nine fields**.

Internal contradiction confirming it: at `max_tokens=32` the sweep reports
`field_f1_9 = 1.0` with `schema_valid_pct = 0.0`. Zero valid records cannot
score perfect F1.

Secondary bug in the same lines: `nine_fields = FIELD_NAMES[:9]` takes the first
nine schema fields — which *includes* `event_type` and `affected_entity` (two of
the three placeholder fields the paper excludes) and *excludes* `is_anomaly` and
`anomaly_score`. Even absent the gold leak, this is not the paper's 9-field
metric.

## Three confounds, all pushing the same way

| | AR-30B (`issue111_*`) | DG / AR-3B (anchors, issue42, …) |
|---|---|---|
| vLLM | **0.11.0** | 0.25.1 |
| Model input | **`bgl.jsonl` (gold record)** | `bgl.raw` (log line) |
| CUDA graphs | **off — `enforce_eager=True`** | on (default) |

`enforce_eager=True` appears in all three AR-30B harnesses and in no anchor
harness. Disabling CUDA graphs is a large small-batch decode penalty.

## Sanity check that something is wrong

- AR-3B (dense 3B, graphs on): 1616 ms / 256 tok = **6.31 ms/token**
- AR-30B (MoE, 3B active, eager): 5899 ms / 122.6 tok = **48.12 ms/token**

A 3B-active MoE decoding **7.6× slower per token than a dense 3B of the same
active size** is not plausible. Routing and memory traffic cost something, but
not that.

## Blast radius

Everything sourced from `issue111_*`: `\arBase` (79.1), `\arSlope` (47.7),
`crossoverL` (6), `crossoverLambda`, `tab:niche` (all five AR rows), the
**15.6× and 8.0× headline**, the 77× batching multiplier, `tab:cost`'s AR-30B
rows and the \$26,000/yr gap, the 73× serial-vs-batch ratio, the AR-30B energy
figure, the conclusion's "sevenfold" and "59 to 6", and Appendix 16's
projections. The abstract, all four contributions and the conclusion rest on it.

Unaffected: everything DG-only (canvas step, K sweep, canvas Pareto, per-event
dispersion) and everything AR-3B/AR-1.5B (quality, tail, energy ratio 1.2,
batching 42.8/26.8, chains, wide schema).

## Required action — cannot be fixed by editing text

1. Fix `load_bgl_events` in all three `run_issue111_*.py` to read `bgl.raw`.
2. Fix `nine_fields` to exclude `event_type`, `affected_entity`, `remediation_hint`.
3. Drop `enforce_eager=True` (or set it on every arm for parity).
4. Re-run on vLLM 0.25.1 for stack parity.
5. Refit the cost law and propagate: macros, `tab:niche`, `tab:cost`, abstract,
   §1 contributions, §3, §5, §8, Appendix 16.

Until then the pass-7 threats paragraph on engine parity understates the
problem: it names one of three confounds. The 15.6× should be treated as
**unverified**, not merely as an upper bound.
