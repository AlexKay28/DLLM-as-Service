# AI Review — v5 (second pass, post-fix)

Re-review after addressing weaknesses from the first pass. Changes made:
- §5 "wins at every" → "wins above the serial crossover" (weakness 4 fixed)
- §3 new "Quality and latency arms" paragraph: inline disclosure + DG p50 CI (weaknesses 1, 3 addressed)
- §5 completeness claim scoped to "unconstrained AR decoding" (weakness 5 addressed)
- §7 footnote: compute resources + licenses (NeurIPS #8, #12 fixed)
- §7 "Base decomposition" paragraph: notes 2-point interpolation, R²=1 trivially (Trend #3 addressed)

## A. SCORES (second pass)

| Dimension | Score | Change | Justification |
|---|---|---|---|
| Soundness | 3→**3.5/4** | ↑ | Quality-arm mismatch now disclosed inline (§3) with conditional framing. DG p50 CI added. Base-decomp R² clarified. Still docked for engine-version asymmetry (GPU work needed). |
| Presentation | 3→**3.5/4** | ↑ | §5 contradiction fixed. Completeness claim scoped. Remaining: three phrasings of "schema-valid" still present (minor). |
| Contribution | **3/4** | = | Unchanged — the contribution itself didn't change, only its framing. |
| Overall | 6→**7/10** | ↑ | Solid accept at a workshop; borderline-accept→accept at a top venue pending the engine-version resolution. |
| Confidence | **4/5** | = | |

## B. WHAT IMPROVED (first → second pass)

| # | Weakness (first pass) | Status | What changed |
|---|---|---|---|
| 1 | Quality arm ≠ latency arm | **Disclosed** | §3 "Quality and latency arms" paragraph now states the mismatch inline, gives the AR-30B F1 (0.870), and frames the advantage as conditional on quality-equivalence. Reviewer no longer has to discover this in §7. |
| 2 | Engine-version asymmetry | **Unresolved** | Still needs GPU re-measurement. The risk is now bounded: the inline §3 disclosure + the §7 footnote make the gap impossible to miss, but the 2.5× headline still rests on vLLM 0.11.0. |
| 3 | No latency CIs | **Partially fixed** | DG p50 CI [379, 380]ms added (bootstrap, n=3200). AR-30B CI is explicitly noted as uncomputed (per-event data not saved). The asymmetry is now disclosed rather than hidden. |
| 4 | §5 contradicts its table | **Fixed** | "wins at every" → "wins above the serial crossover ($L^*{=}48$)". No longer contradicts the 0.7× row. |
| 5 | Grammar-constrained AR open | **Scoped** | Completeness advantage now explicitly scoped to "unconstrained AR decoding" in §5, with the grammar-constrained gap acknowledged inline. |

## C. NEURIPS CHECKLIST (second pass)

| # | Question | Before | After | Change |
|---|---|---|---|---|
| 7 | Error bars / significance | partial | **yes** | DG p50 CI added; AR-30B gap disclosed |
| 8 | Compute resources | partial | **yes** | Footnote: GPU type, stack versions, ~20 GPU-hours |
| 12 | Licenses | partial | **yes** | Footnote: all asset licenses stated |
| Others | — | — | unchanged | |

## D. REMAINING ISSUES (not fixable without GPU)

1. **Engine-version asymmetry** (weakness 2). The AR-30B ran on vLLM 0.11.0; everything else on 0.25.1. The 2.5× headline, 25× batching, and cost table rest on the 0.11 number. This is now thoroughly disclosed (§3 inline, §7 footnote) but not resolved. **Requires: GPU re-measurement of AR-30B on vLLM 0.25.1.**

2. **Grammar-constrained AR** (weakness 5). The completeness advantage is now scoped to unconstrained AR, which is honest. But a reviewer may still ask whether the advantage survives grammar-constrained decoding. **Requires: GPU run of outlines/xgrammar on vLLM 0.25 with AR-3B.**

3. **"Schema-valid / schema validity / schema completeness"** — three phrasings for related concepts. Not a correctness issue but a clarity nit. Minor.

## E. TROUBLING-TRENDS AUDIT (second pass)

| Trend | Before | After | Change |
|---|---|---|---|
| #1 Explanation vs speculation | LOW | LOW | Unchanged. |
| #2 Unidentified gains | MODERATE | MODERATE | Unchanged — the advantage is still attributed to "the staircase shape" collectively, not decomposed per mechanism. The base-decomposition paragraph helps but doesn't fully isolate which mechanism (parallelism vs K-step denoising vs canvas size) drives the 2.5×. |
| #3 Mathiness | LOW | **NONE** | Base-decomp R² concern resolved: the paper now explicitly states it's a 2-point interpolation, not a regression, so R²=1 trivially. |
| #4 Language misuse | LOW-MODERATE | LOW | "Natural output length" glossed in abstract (v5). "Schema-valid" phasings remain (minor). |

## F. UPDATED TOP-3 (ranked by remaining impact)

1. **Re-measure AR-30B on vLLM 0.25.1** (GPU, ~1 day). The only remaining soundness risk. Everything else is disclosed; this is the one unresolved number.
2. **Run grammar-constrained AR on a current engine** (GPU, ~2 hours). Would either bound or remove the completeness advantage.
3. **Decompose the 2.5× advantage by mechanism** (CPU analysis of existing data). Which of parallelism-within-canvas, K-step denoising, or canvas size drives the gap? The base-decomposition and K-sweep data partially answer this but aren't framed as a mechanism attribution.

## G. SUMMARY

The second pass is meaningfully stronger. Four of five weaknesses are addressed at the text level (disclosure, scoping, CIs, contradiction fix). The fifth (engine version) is now impossible for a reviewer to miss but requires GPU work to resolve. The NeurIPS checklist moved from 3 "partial" items to 0. The troubling-trends audit dropped from one MODERATE+LOW-MODERATE to one MODERATE+LOW. The paper is now in shape for a workshop submission; for a top venue, the engine-version re-measurement is the gating item.
