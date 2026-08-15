# Figures

Nine figures, one visual system. Built by `paper/figures/make_figures.py`
against `results/*.json`; the shared style lives in `paper/figures/style.py`.
Each builder function prints the values it plotted so they can be diffed
against `paper/macros.tex` and `paper/NUMBERS.md`.

```bash
cd paper/figures && python3 make_figures.py
```

Output is PDF (for print) and PNG (for review) side by side under
`paper/figures/`.

---

## The set

| # | File | Carries | Data source |
|---|---|---|---|
| 1 | `fig1_cost_shapes` | The line-vs-staircase thesis and the $L^*\approx59$ crossover | `p2_ar3b_L*.json`, `appD_ar3b_L*.json`, `issue68_dg_L*.json` |
| 2 | `fig2_canvas_edges` | Sharp edges at canvas boundaries; all seven steps are the same step | `issue68_dg_L*.json` |
| 3 | `fig3_input_sweep` | The second variable — DG's prefill slope is 2.6× AR's | `issue44_{ar3b,dg}_input_sweep.json` |
| 4 | `fig4_ksweep` | Base decomposition: latency saturates, quality collapses below K=4 | `issue44_dg_ksweep_K*.json` |
| 5 | `fig5_quality_vs_length` | The truncation regime, where AR degrades gracefully and DG does not | `issue43_nine_field_f1.json`, `issue43/*.json` |
| 6 | `fig6_batching` | The ~80× vs ~1.2× gap, and the prefix-cache test that only half explains it | `issue42_batching.json`, `llada_dinfer/batch_sweep.json`, `appD_analysis.json` |
| 7 | `fig7_decision_map` | The decision procedure as a map of the $(L,\lambda)$ plane | derived from the fitted laws |
| 8 | `fig8_defects` | Five harness defects, before → after, direction of bias | `CHECKLIST.md` |
| 9 | `fig9_deployment` | The quality-matched pair: latency at forced $L$, and standing cost | `issue_evidence_ar1b_*.json`, `issue68_dg_L*.json`, `issue_energy_measurement.json` |

---

## House rules

These are enforced by `style.py` and by review, not by a linter.

1. **One colour per model, everywhere.** AR = blue `#2a78d6`, DG = orange
   `#eb6834`, LLaDA = aqua `#1baf7a` — categorical slots 1/2/3 of the
   validated default data-viz palette. On a white surface with `--pairs all`:
   worst CVD ΔE 9.2, worst normal-vision ΔE 24.0, all checks pass. Aqua is
   below 3:1 contrast on white, so it always carries a direct label.
2. **Greyscale-safe.** Every series also differs in marker shape, so the set
   survives a black-and-white print.
3. **No dual y-axes, ever.** Two measures of different scale become two
   stacked panels sharing an x (see `fig4_ksweep`).
4. **No mixed units on one scale.** `fig8_defects` puts each defect on its own
   before → after track with its unit labelled; `fig9_deployment` panel B
   scales each pair to its own maximum and prints the raw values.
5. **Recessive chrome.** Solid hairline grid one shade off the surface, no
   dashed gridlines, no top/right spines, generous padding.
6. **Direct labels over legends** where two series can be labelled at their
   endpoints; a legend is still present whenever there are two or more series.
7. **Units on every axis.** Latency in ms, rate in events/s, length in tokens.
8. **Provenance in the caption**, not in the figure. Every caption names the
   artifact glob, `n` per cell, and the configuration ($K$, canvas, batch).
9. **Lower bounds are marked.** The DiffusionGemma batching bars are hatched
   because the engine caps their batch at 8.

## Consistency check

`make_figures.py` prints every plotted value on stdout. Two of them are
asserted against artifacts inline (`fig9` re-reads the AR-1.5B forced-length
p50s and fails if they drift). The rest are diffed by eye against
`paper/macros.tex`, whose comments name the artifact each number came from.
