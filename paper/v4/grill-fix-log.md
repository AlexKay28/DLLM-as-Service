# Grill Fix Log — 2026-08-16

## Pass 6 (27-check skill, full audit of paper/v4)

Note: `paper/v4` was being edited concurrently by another agent session
during this pass (HEAD advanced ~6 commits mid-audit; figures were swapped
to `_hires.png` and §2 was restructured into subsections). All fixes below
were applied on top of that moving state and verified against a clean
recompile afterwards.

### CRITICAL

| # | File | Change | Check | Severity |
|---|------|--------|-------|----------|
| 1 | sections/01_introduction.tex | "The plot twist": direction of the harness bias was inverted — said diffusion first looked **slower** by 4×; the artifacts say the crippled harness made diffusion look **faster** by 4× (AR-1.5B 1841.6 ms on HF vs DG 420.2 ms). Rewrote so the corrections *remove* a pro-diffusion bias, and attributed the surviving 15× to the MoE-matched baseline switch rather than to the corrections. | Check 2, 5, 22 | CRITICAL |
| 2 | sections/04_deployment_losses.tex | "ten **million** events per day" → "ten **thousand**". $5/day, $76/day and the $26,000 annual gap only reconcile at 10k events/day (10M/day would be $5,000 vs $76,000/day and needs ~683 GPUs, contradicting "on a single GPU"). | Check 1, 12 | CRITICAL |
| 3 | sections/03_deployment.tex | `tab:niche` was labelled "measured … $n{=}50$" but its DG column is Eq.~\ref{eq:dg} evaluated per canvas count (379/379/379/498/737 = the law exactly); no DG $n{=}50$ sweep exists. Caption now states AR measured / DG law-derived, gives the forced measured DG values (501 ms, 739 ms), and discloses that the two columns compare AR-at-EOS against DG-at-full-canvas. Body text updated to match. | Check 2 | CRITICAL |
| 4 | assets/macros.tex | `cacheOnMult` (42.8) / `cacheOffMult` (26.8) comments claimed "AR-30B"; `results/issue42_batching.json` records `Qwen/Qwen2.5-3B-Instruct`. Comments corrected to "dense AR-3B" + note that the AR-30B MoE multiplier is 77×. | Check 17 | CRITICAL |

### IMPORTANT

| # | File | Change | Check | Severity |
|---|------|--------|-------|----------|
| 5 | sections/01_introduction.tex | `tab:summary` put the AR-3B max/median (1.03×) under an "AR-30B (MoE)" header with no disclosure; every other AR cell is AR-30B. Caption now defines max/median and names the per-row model. | Check 13, 24 | IMPORTANT |
| 6 | sections/05_survives.tex | `fig:chains` panel B caption had the mechanism backwards: "the tail advantage disappears with chain length". Data (`chain_analysis.json`, `P_E2`) shows the AR/DG ratio at p99.9 *rises* 1.04× → ~1.9× with depth, i.e. DG's tail **penalty** averages out and DG **keeps more** of its advantage. Also relabelled the plotted quantity (AR/DG ratio at p99.9, not "DG's p99.9/p50"). | Check 7, 26, 27 | IMPORTANT |
| 7 | sections/02_cost_laws.tex | "The DG base is roughly **half** prefill" contradicted Appendix 10's 38% (and its own ~150 ms floor: 142.7/378.8 = 38%). → "roughly two-fifths ($\approx$38%)". | Check 8 | IMPORTANT |
| 8 | sections/03_deployment.tex | "CV" used bare in `tab:contrast` (§3); expanded only in §4. Expanded on first use. | Check 23 | IMPORTANT |
| 9 | sections/11_appendix_batching.tex, 17_appendix_batching_arch.tex | 42.8×/26.8× and "AR's 26–43×" read as the AR-30B MoE; both are the dense AR-3B. Disclosed in both, contrasted against the AR-30B's 77×. | Check 13, 19 | IMPORTANT |
| 10 | sections/02_cost_laws.tex | `fig:edges` claimed the seven canvas transitions "cost the same within noise" with the supporting range no longer in the paper. Restored the measured range (113.1–122.6 ms, median 119.5, 8% spread) — the spread exceeds p50 repeat noise (CV 0.25%), so "within noise" was not defensible unqualified. | Check 2, 27 | IMPORTANT |
| 11 | sections/02_cost_laws.tex | Generalisation subsection asserted "within 2%" with no numbers. Restored `bglStep` 118.5 vs `hdfsStep` 120.5 (1.7%). | Check 24 | IMPORTANT |
| 12 | sections/05_survives.tex | "roughly fourteen times faster per hop" silently uses the chain-harness DG p50 (419 ms), not the 379 ms anchor used everywhere else. Source and reason now stated. Also "advantage compounds" → "gap widens in absolute terms" (the ratio is constant with depth). | Check 1, 24 | IMPORTANT |
| 13 | 5 section files | 16 unreferenced `\label`s: appendices `app:forced`, `app:quality`, `app:input_canvas` were never pointed to from anywhere (orphaned payloads), and 10 figures + 2 tables had no `\ref`. Added textual references for all except `eq:lcross`. | Check 3, 22 | IMPORTANT |
| 14 | assets/macros.tex | 5 dead macros (`gpuModel`, `hdfsStep`, `bglStep`, `dgStepLo`, `dgStepHi`) — orphaned by earlier prose pruning. Rather than delete, restored all five to the text (items 10, 11 above and the intro stack line), which also repaired the two unquantified claims. Dead macros now 0. | Check 21 | IMPORTANT |

### MINOR

| # | File | Change | Check | Severity |
|---|------|--------|-------|----------|
| 15 | assets/macros.tex | `dgStepLo` comment said the 113.1 ms step was `1536->1792`; it is `1280->1536` (1536→1792 is 117.5). | Check 17 | MINOR |

## Verified correct — no change needed

- AR-30B cost law: intercept 79.1, slope 47.68→47.7, $R^2$ 1.0, crossover 6.0 — all match `issue111_qwen3_serial_sweep.json`.
- `tab:niche` AR column (1604/5923/5899/5898/5910) and actual-out (32/121/123/123/123) match the JSON p50s and `mean_output_tokens` exactly.
- Crossover integer-safety: at $L{=}6$ AR is faster (365.3 vs 378.8), at $L{=}7$ DG is (413.0 vs 378.8). Paper says "$L \geq 7$" and "above $L^{*}$" throughout — correct.
- `tab:cost`: every events/hr and $/event recomputes to <1%. $R_{cost}=1.8$, 16×5408=86,528 correct.
- Appendix 16 projection table (13435/498/27.0, 26791/618/43.4, 53503/857/62.4) reproduces the cost laws exactly.
- Conclusion's "sevenfold" (47.7/6.29 = 7.6) and "crossover from 59 to 6" ((378.8−7.27)/6.29 = 59.1) both check out.
- Per-event attribution (Check 19): `ar3b_L256_r1_preds.jsonl` median 1616 ms = dense AR-3B, CV 0.17%→0.2, max/med 1.03; `dg_L256_r1` median 377 ms, CV 20.96%→20.9, max/med 3.21. Macros and Appendix 15 label these correctly.
- `dgStep` = 119.50 = exact median of the 7 measured canvas transitions.
- Energy: 10.1 J / 4.2 J / 5.1 J and idle 92.7/58.6/52.6 W all match their JSONs; the three-idle-baseline caveat is disclosed in both §3 and §7.
- Canvas Pareto: 368.5 ms (−2.0% vs 375.98), F1 0.8644→0.864, canvas-64 +48% — all correct.
- Abstract / contributions / conclusion agree on 15×, 8×, $L^{*}{=}6$, 77×, $\leq$1.7× (Check 14).

## False positives skipped (from catalog)

| Pattern | Check | Why skipped |
|---------|-------|-------------|
| `align` blocks without a `where` clause | Check 24 | Variables defined in the preceding `equation`'s where-block. |
| FRE 62–72 in §2, App 10/13/15 | Check 23 | Deliberate outcome of the storytelling passes; those sections are 2–7 sentences, so the score is not meaningful. Median sentence length is 13–23 w everywhere and there are 0 sentences >60 w. |
| §3 "narrows with output length" vs App 16 "advantage grows" | Check 5 | Different regimes (AR at natural EOS vs AR forced long); both state their mechanism. |
| DG events/hr 9504 vs 9499 | Check 12 | 378.8 vs 379 rounding, 0.05%. |
| AR-1.5B $0.000854 vs $0.0009 | Check 12 | 4-dp rounding. |

## Results

- Compilation: **0 errors, 0 overfull, 0 undefined**, 20 pages
- `verify_numbers.py`: **0 errors, 0 precision violations** (112 pre-existing warnings, unchanged)
- Dead macros: **0** (was 5)
- Unreferenced fig/tab/app labels: **0** (was 16); `eq:lcross` left unreferenced by design
- Derived-value mismatches: **0** across 12 recomputed quantities
