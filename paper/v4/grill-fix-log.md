# Grill Fix Log — 2026-08-16

## Pass 8 (37-check audit)

### CRITICAL findings

| # | Check | Finding | Severity | Fix |
|---|-------|---------|----------|-----|
| 1 | 1/20 | `\arBase`=79.1 but artifact=34.6; `\arSlope`=47.7 but artifact=7.2; `\crossoverL`=6 but artifact=48.0 — corrected serial sweep exists but macros never updated | CRITICAL | NOT FIXED — requires #113 re-measurement; threats section already discloses |
| 2 | 31 | Abstract/intro/conclusion state "fifteen times faster" and "crossover at 6 tokens" as fact; threats says "unverified"; corrected artifact shows 2.4× and 48 tokens | CRITICAL | NOT FIXED — depends on #113; threats section discloses |
| 3 | 30 | `03_deployment.tex` (384 lines) contains 100% duplicate content from `04_deployment_losses.tex` (120 lines) and `05_survives.tex` (134 lines). All three `\input`'d — paper renders same text twice | CRITICAL | FIXED — 03 is the superset; 04 and 05 are now empty stubs that `\input` the relevant subsection from 03 |
| 4 | 34 | `\cite{qwen25}` on Qwen3-30B-A3B — wrong citation; no `qwen3` entry in references.bib | IMPORTANT | NOT FIXED — need to add Qwen3 citation to references.bib |
| 5 | 36 | `01_introduction.tex` cost-law table: "Per-canvas cost" row for AR contains only `,` — degenerate cell | IMPORTANT | FIXED — replaced with `—` (em-dash: not applicable) |
| 6 | 25 (new) | Intro uses bare "AR" and "DG" without expanding on first use in section | IMPORTANT | FIXED — first occurrence in intro now spells out "autoregressive (AR)" and "block-diffusion (DG)" |
| 7 | 28 | Abstract Flesch=13.9 (FAIL, threshold 18-55); bracket density=1.6 (FAIL, threshold ≤1.5) | IMPORTANT | NOT FIXED — needs prose rewrite to reduce density |
| 8 | 32 | Quality F1 from dense AR-3B, latency from AR-30B MoE — disclosed in threats but not at point of comparison | IMPORTANT | NOT FIXED — requires restructuring the comparison |
| 9 | 2 | `issue111_ar30b_energy.json` has no provenance block, F1=0.3167 — old broken harness | IMPORTANT | NOT FIXED — requires #113 re-measurement |
| 10 | 15 | 1 overfull vbox (3.47pt) | MINOR | NOT FIXED — negligible |

### OK findings (no issues)

- Check 16: All derived calculations recompute correctly (crossover, energy ratio, cost-per-event, marginal ratio)
- Check 18: Crossover integer-safe — "≥ 7 tokens" is correct for the macro value of 6
- Check 21: 0 dead macros (47 defined, 47 used)
- Check 12: Cost-per-event table arithmetic correct (events/hr, $/event, duty cycles)
- Check 15: Compilation clean (0 errors, 0 undefined, 1 overfull)

## Compilation: 0 errors, 1 overfull, 0 undefined
## Dead macros: 0
## verify_numbers.py: 4 errors (all AR-30B staleness — #113 blocker)
