# Paper Criteria

What "done" looks like for this paper. Four categories — **successful,
accurate, interesting, readable** — each with measurable checks. Most checks
are enforced by `paper/verify_numbers.py` and `paper/eval_readability.py`
(both must exit 0); the rest are manual review points flagged at the end of
each section.

This file is version-independent: it applies to `paper/v5/` and every later
snapshot. The two scripts take `--version <dir>` (default `v5`).

---

## 1. Successful

A paper succeeds when a reader can reconstruct the claim, the comparison, and
the regime from the text alone, and every cited number traces to a committed
artifact.

| # | Check | How measured |
|---|---|---|
| S1 | Abstract states: the claim, the comparison (arms matched on what), the headline number, and the regime (five conditions). | Manual / `eval_readability.py` abstract-number count. |
| S2 | Every contribution in §1 maps to a section that delivers it, and §8 restates the regime + five conditions. | Manual. |
| S3 | Every number cited in prose or a macro traces to a result JSON with a `provenance` block. | `verify_numbers.py` (`check_provenance`, `check_macro_values`). |
| S4 | The five CHECKLIST defect magnitudes match their artifacts. | `verify_numbers.py` (`check_defect_magnitudes`). |
| S5 | Cost-law macros (`arBase`, `arSlope`, `crossoverL`) equal their artifact fields within tolerance. | `verify_numbers.py` (`check_macro_values`). |
| S6 | Paper compiles: 0 errors, 0 undefined refs, 0 overfull boxes. | `pdflatex` + log parse (Makefile `paper` recipe). |

---

## 2. Accurate

A paper is accurate when no sentence asserts the opposite of its own data, and
no comparison silently switches which arm it is measuring.

| # | Check | How measured |
|---|---|---|
| A1 | §1 must not claim the pre-correction advantage "survived" — the 15.6× was the *defective* harness; corrections brought it to 2.5×. | `verify_numbers.py` (`check_no_backwards_fifteenfold`). |
| A2 | A bold paragraph header must not contradict the table below it (e.g. "wins at every length" over a table with a 0.7× row). | `verify_numbers.py` (`check_header_table_consistency`). |
| A3 | Every latency/quality ratio in §3/§5 names its arm (AR-30B, dense AR-3B, AR-1.5B, LLaDA, or DG) or cites a table/figure that does. | `verify_numbers.py` (`check_arm_attribution_inline`). |
| A4 | Macro values match artifact values (no drift after re-measurement). | `verify_numbers.py` (`check_macro_values`). |
| A5 | Table headers match their rows (column count, arm labels, units). | Manual / `verify_numbers.py` (`check_duty_column_defined`). |
| A6 | The fact that the quality arm ≠ latency arm (dense AR-3B vs AR-30B MoE) is disclosed inline at first use, not only in §7 Threats. | `verify_numbers.py` (`check_arm_attribution_inline`). |
| A7 | Numeric precision policy holds (latency 1 dp, F1 3 dp, ratios 1 dp, rates 3 dp). | `verify_numbers.py` (`check_precision_policy`). |
| A8 | Engine-version asymmetry (vLLM 0.11 vs 0.25) is disclosed in §7, not hidden. | Manual (grill C1 precedent). |

---

## 3. Interesting

A paper is interesting when the stakes are concrete and the punchlines are not
buried.

| # | Check | How measured |
|---|---|---|
| I1 | §1 Data paragraph shows one BGL log line in → one JSON record out, with the 9 scored vs 3 excluded fields marked. | `verify_numbers.py` (`check_data_sample_present`). |
| I2 | The task's stakes are named concretely (SOC / driving / trading), not abstractly. | Manual. |
| I3 | The tail-latency punchline ("3.21× breaks a batch-1 SLA") appears in §3 Results, not only in §4 Limits. | `verify_numbers.py` (`check_tail_punchline_in_results`). |
| I4 | One canonical ratio per claim; cross-harness variants demoted to a footnote or parenthetical, not three competing numbers in two paragraphs. | `verify_numbers.py` (`check_canonical_ratio`). |
| I5 | The input-length caveat (DG prefill 2.6× steeper, moves $L^{*}$) is in §2 or §3, not buried in §4. | `verify_numbers.py` (`check_input_length_not_buried`). |

---

## 4. Readable

A paper is readable when a first-time reader can follow it without re-reading
sentences or hunting for definitions.

| # | Check | How measured |
|---|---|---|
| R1 | Flesch Reading Ease in [18, 55] for main text, [20, 80] for appendices. | `eval_readability.py` (`check_readability`). |
| R2 | Median sentence length in [10, 28] words. | `eval_readability.py`. |
| R3 | ≤ 2 sentences > 40 words per section; 0 sentences > 60 words. | `eval_readability.py`. |
| R4 | Acronyms expanded on first use: SLA, CV, SOC, EOS, KV, HDFS, B2B, MoE. | `verify_numbers.py` (`check_acronyms_expanded`). |
| R5 | No over-explained definitions (e.g. "a ratio above 1 means X" immediately after defining $R = X/Y$). | `eval_readability.py` (`check_overexplained_definitions`). |
| R6 | Repeated caveats ("dense AR-3B not AR-30B", "engine-bounded") consolidated — appear inline at most once outside §7. | `eval_readability.py` (`check_repeated_caveats`). |
| R7 | Figure captions are self-contained: each states what is shown and the finding, with arm + n disclosed. | `eval_readability.py` (`check_caption_self_contained`). |
| R8 | Number density ≤ 4.0 / 100 words in main text; bracket density ≤ 1.5 / 100 words. | `eval_readability.py` (`check_density`). |
| R9 | §1 "plot twist" is neutral prose or removed; no editorializing ("the arm we were not rooting for"). | `verify_numbers.py` (`check_plot_twist_neutral`). |

---

## How to run

```bash
# Accuracy + success (reads .tex + results/)
python3 paper/verify_numbers.py

# Readability + density (reads paper/v5/ by default)
python3 paper/eval_readability.py
python3 paper/eval_readability.py --version v4   # compare against prior version

# Both must exit 0 before a snapshot is publishable.
```

## Manual review points (not automated)

- S1/S2: abstract↔contributions↔conclusion alignment.
- I2: stakes named concretely.
- A8: engine-version disclosure present in §7.
- Figure content (captions are checked; the rendered PNGs are not).
