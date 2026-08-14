# Figure plan

Print targets for the paper. Every figure lists its data source, what it must
show, and the failure mode to avoid. **All numbers must reconcile with
`paper/NUMBERS.md`** — a chart that disagrees with the text is worse than no
chart.

Existing PNGs in `results/` were made for issue comments (screen, wide aspect,
default styling). They are **drafts, not print figures** — every one needs a
rebuild at consistent size, font, and palette.

When building these, load the `dataviz` skill first so the whole set reads as
one system.

---

## Priority order

| # | Figure | Exists? | Priority |
|---|---|---|---|
| 3 | Quality vs output length | **No** | **Highest — the paper's strongest result has no chart** |
| 1 | Joint λ×L grid | draft | Highest — this is Figure 1 |
| 2 | Cost laws | draft | High |
| 4 | Batching multipliers | **No** | High — the mechanism section's only visual |
| 8 | Per-field corruption at L=512 | **No** | High — the mechanism behind Fig. 3 |
| 5 | Chain tail convergence | draft | Medium |
| 6 | Perturbation refutation | **No** | Medium |
| 7 | Wide schema | draft | Low — may cut for space |

---

## Figure 3 — Quality vs output length (**build this first**)

**The paper's strongest single result, currently unillustrated.**

- **Data:** `results/p2_{dg,ar3b}_L{32,64,128,256,512}.json`,
  `results/appD_{dg,ar3b}_L{1024,2048}.json` → `field_f1`, `schema_valid_pct`
- **Numbers:** `NUMBERS.md` §3

**Must show:** Field-F1 on y, forced output length L on x (log2: 32…2048), one
line per model. Diffusion is **flat** at 0.795/0.789/0.790 across 512/1024/2048;
AR **decays monotonically** 0.599/0.495/0.473. The two lines *cross twice* —
that is the whole point.

**Must show the three regimes** as shaded bands with labels:
- **Truncation** (L ≤ 128) — both collapse, but AR degrades gracefully
  (0.654 vs 0.329 at L=128). *An AR advantage.*
- **Matched** (L = 256) — AR modestly ahead (0.826 vs 0.787). Mark the task's
  natural output (~176 tokens) with a vertical rule.
- **Over-generation** (L ≥ 512) — diffusion flat, AR decaying.

**Second panel:** `schema_valid_pct` on the same x. It carries the truncation
story (diffusion 3.3% vs AR 66.7% at L=128; both ~100% from L=256 up) and
explains why F1 collapses at small L for structural rather than semantic reasons.

**Failure mode to avoid:** cropping to L ≥ 256 so diffusion looks uniformly
better. The truncation regime is a genuine AR win and must be visible. A
reviewer who finds it omitted will assume it was hidden.

---

## Figure 1 — Joint rate × length grid (the headline figure)

- **Data:** `results/rate_lag_{dg,ar3b}_L{64,128,256,512}.json`
- **Draft:** `results/joint_rate_lag.png` / `.html`
- **Numbers:** `NUMBERS.md` §4

**Must show:** two orthogonal walls. Diffusion lag flat in L, walling in λ
(knee 2.0–2.5 ev/s); AR lag flat in λ, walling in L (knee 64–96 ev/s). A reader
should see "these fail along perpendicular axes" in under three seconds.

Two candidate forms — try both:
1. Small-multiples: lag-vs-λ, one panel per L, both models per panel.
2. Heatmap of log(lag ratio AR/DG) over the λ×L grid, diverging palette centred
   at ratio 1, with the win-region boundary drawn.

Form 2 is likelier to be the real Figure 1; form 1 is the honest fallback since
it shows absolute values rather than only a ratio. Mark the crossovers
(λ ≈ 1.8 ev/s, L ≈ 65 tokens) explicitly.

**Caption must disclose** the serving asymmetry: DiffusionGemma at
`--max-num-seqs 1` vs AR at the default 256-way scheduler, forced by
`vllm-project/vllm#50699`. Not a footnote — in the caption.

---

## Figure 2 — Cost laws

- **Data:** same L-sweep files as Fig. 3, `p50_ms`; fits from
  `results/appD_analysis.json`
- **Draft:** `results/appD_costlaws.png`
- **Numbers:** `NUMBERS.md` §2

**Must show:** measured points plus fitted curves — AR linear
(`7.27 + 6.29·L`), diffusion stepwise (`418.24 + 113.43·(⌈L/256⌉−1)`). The
**staircase** is the signal: keep it visible, do not smooth it. Mark the
crossover at L ≈ 65 and annotate the L=1024/2048 extrapolation checks (+8.4%,
+11.7%).

**Inset or second panel:** ratio AR/DG vs L, showing saturation toward 14.2×.

⚠ **Blocked on reconciling the fit discrepancy** (`NUMBERS.md` §2.1):
`appD_analysis.json` and `appD_report.md` disagree on the AR intercept/slope/σ
and on the DG step. Resolve before drawing fitted lines, or the figure will
contradict whichever source the text cites.

---

## Figure 4 — Batching multipliers (the mechanism)

- **Data:** `results/appD_analysis.json` → `batching_multipliers`;
  dInfer 1.18× from `results/llada_dinfer/llada_dinfer_report.md`
- **Numbers:** `NUMBERS.md` §5

**Must show:** batch-1 → peak throughput multiplier for five configurations, on
a **log y-axis** (values span 1.18× to 86×). Group by architecture, not runtime,
so the reader sees the split is diffusion-vs-AR rather than vLLM-vs-dInfer:

| | multiplier |
|---|---|
| DiffusionGemma-26B (vLLM), L=128 | 1.7× |
| DiffusionGemma-26B (vLLM), L=512 | 1.5× |
| LLaDA-8B (dInfer) | 1.18× |
| Qwen2.5-3B (vLLM), L=128 | 77.1× |
| Qwen2.5-3B (vLLM), L=512 | 86.0× |

**The dInfer bar is the load-bearing one** — a second runtime, a different model
family (dense vs MoE), same cap. Annotate it as the control; it is what makes
this architectural rather than a vLLM bug report.

**Must annotate honestly:** the DiffusionGemma bars were measured with batch
capped at 8 by the engine crash, so they are **lower bounds**. Hatch or
asterisk them.

---

## Figure 8 — Per-field corruption at L=512

- **Data:** `results/p2_{dg,ar3b}_L512.json` → `per_field_f1`
- **Numbers:** `NUMBERS.md` §3

**Must show:** paired bars per field, sorted by AR's degradation. The story is
that AR corrupts fields it should copy **verbatim** — `host` 0.673 vs 1.000,
`component` 0.763 vs 0.990, `severity` 0.617 vs 0.993, `is_anomaly` 0.673 vs
0.993. Separate the structural (copy) fields from the semantic (inference)
fields visually; the effect is concentrated in the copy fields, which is exactly
what makes it surprising.

Small and cheap to build, and it supplies the *mechanism* behind Figure 3.
Strongly consider placing them side by side.

---

## Figure 5 — Chain tail convergence

- **Data:** `results/chain_analysis.json` → `P_E2_tail_concentration`
- **Draft:** `results/chain_clt.png`

**Must show:** p99.9 ratio (AR/DG) vs chain depth, rising from 1.041 at D=1 to
1.77–2.07 at D≥3, with a horizontal reference line at the mean p50 ratio (1.93)
that it converges to. Optionally overlay the analytical CLT prediction.

**Do not** imply a quality win: annotate that AR's Field-F1 is ~0.04 higher at
every depth, and that both models drop sharply from D=1 to D=3. This is a
latency result only.

---

## Figure 6 — Perturbation refutation

- **Data:** `results/perturbation_report.md`, `results/perturbation_{ar3b,dm_hf}.json`
- **Numbers:** `NUMBERS.md` §7.1

**Must show:** F1 drop per category at intensity 1.0, AR vs diffusion, 10
categories sorted by the ratio. Diffusion is worse in **10/10** — the visual
should make the sweep unmistakable, since a reader skimming will assume a mixed
result.

Mark the two categories where AR is exactly flat (whitespace, abbreviation) —
they are the most striking cells and the ratio is undefined there, so plot the
raw drops rather than the ratio.

---

## Figure 7 — Wide schema (optional, cut first for space)

- **Data:** `results/wide_schema_analysis.json`
- **Draft:** `results/wide_schema.png`

Latency ratio and consistency vs field count. **Must show that AR wins
consistency at every width** (96.5% vs 87.5% at F=100) — the draft chart's
framing overstates the diffusion case. Also annotate that diffusion emits more
tokens at every width (1406 vs 951 at F=100), so the latency ratio is not a
like-for-like comparison.

---

## Cross-cutting requirements

1. **Consistent model naming** across every figure — pick one form
   (`DiffusionGemma-26B` / `Qwen2.5-3B` / `LLaDA-8B`) and never vary it.
2. **Consistent colour per model** across all figures — a reader must not have
   to re-read the legend.
3. **Campaign provenance in every caption.** Three different hosts and stacks
   are in play (`NUMBERS.md` §0); a figure mixing campaigns without saying so is
   a defect.
4. **No dual y-axes** unless unavoidable; prefer stacked panels sharing an x.
5. **Greyscale-safe** — differentiate by marker and line style, not colour alone.
6. **Every axis labelled with units.** Latency in ms, rate in events/s, length in
   tokens.
7. **Single-run data.** No error bars exist because no cell was repeated. Either
   state "single run, n=300 per cell" in every caption, or run repeats first.
   Silently omitting error bars invites the question.
8. Emit **PDF or SVG** for print, not PNG. Regenerate the drafts rather than
   upscaling them.

## Suggested build

`paper/figures/` for scripts (`fig1_joint_grid.py`, …), output to
`paper/figures/out/`. Each script reads `results/*.json` directly so figures
regenerate from source, and prints the values it plotted so they can be diffed
against `NUMBERS.md`.
