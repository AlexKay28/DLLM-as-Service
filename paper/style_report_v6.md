# Style Report — v6

Generated: 2026-08-22T17:00:55.278091

## Reference Corpus Baseline

| Metric | Reference Corpus |
|---|---|
| Words | 141,541 |
| Sentences | 5,117 |
| Sentence median (words) | 21 |
| Sentence p90 (words) | 45 |
| Sentences >40 words (%) | 13.0% |
| Metaphors found | 6 |
| Number density (per 100 words) | 9.6 |

## Section Results

| Section | Words | Sent | Med | p90 | >40% | Metaphor | JSD | Vocab | Delta | TF-IDF | FuncWord | SentLen | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_introduction | 885 | 43 | 19 | 33 | 2.3% | 2 | 0.993 | 0.768 | 0.7546 | 0.6175 | 0.8897 | 0.5456 | FAIL |
| 02_cost_laws | 703 | 33 | 18 | 39 | 9.1% | 3 | 0.996 | 0.805 | 0.6477 | 0.598 | 0.7906 | 0.5609 | FAIL |
| 03_deployment | 717 | 29 | 24 | 38 | 6.9% | 2 | 0.998 | 0.801 | 0.7022 | 0.4547 | 0.7961 | 0.4731 | FAIL |
| 04_deployment_losses | 447 | 21 | 17 | 38 | 4.8% | 0 | 0.998 | 0.788 | 0.7426 | 0.4545 | 0.8082 | 0.4062 | PASS |
| 05_survives | 940 | 46 | 19 | 37 | 4.3% | 5 | 0.996 | 0.769 | 0.6764 | 0.5236 | 0.8513 | 0.636 | FAIL |
| 06_related_work | 233 | 14 | 13 | 30 | 0.0% | 0 | 0.996 | 0.795 | 0.6442 | 0.4843 | 0.8513 | 0.255 | PASS |
| 07_threats | 1139 | 56 | 21 | 32 | 0.0% | 0 | 0.996 | 0.811 | 0.6577 | 0.6142 | 0.834 | 0.6883 | FAIL |
| 08_conclusion | 270 | 15 | 17 | 38 | 0.0% | 2 | 0.995 | 0.798 | 0.7823 | 0.5029 | 0.8046 | 0.2617 | FAIL |
| 09_appendix_defects | 695 | 45 | 12 | 27 | 4.4% | 0 | 1.0 | 0.728 | 0.7398 | 0.5076 | 0.8149 | 0.5077 | PASS |
| 10_appendix_decomp | 111 | 5 | 27 | 37 | 0.0% | 0 | 1.0 | 0.756 | 0.765 | 0.3898 | 0.8065 | 0.1046 | PASS |
| 11_appendix_batching | 116 | 5 | 23 | 39 | 0.0% | 0 | 1.0 | 0.755 | 0.7337 | 0.4473 | 0.721 | 0.0836 | PASS |
| 12_appendix_refuted | 117 | 7 | 14 | 33 | 0.0% | 0 | 0.999 | 0.788 | 0.7398 | 0.4515 | 0.7755 | 0.1044 | PASS |
| 13_appendix_input_canvas | 248 | 12 | 22 | 37 | 0.0% | 1 | 0.999 | 0.8 | 0.7248 | 0.4592 | 0.7527 | 0.2439 | PASS |
| 14_appendix_quality | 49 | 3 | 17 | 19 | 0.0% | 0 | 0.999 | 0.84 | 0.7189 | 0.2507 | 0.5632 | 0.1069 | PASS |
| 15_appendix_distributions | 57 | 3 | 22 | 29 | 0.0% | 0 | 1.0 | 0.731 | 0.7413 | 0.1712 | 0.5222 | 0.0715 | PASS |
| 16_appendix_forced | 184 | 7 | 22 | 57 | 14.3% | 0 | 0.999 | 0.768 | 0.6486 | 0.4426 | 0.7339 | 0.1558 | FAIL |
| 17_appendix_batching_arch | 160 | 6 | 26 | 44 | 16.7% | 0 | 0.994 | 0.821 | 0.6802 | 0.5091 | 0.7582 | 0.1591 | PASS |
| 18_appendix_humaneval | 364 | 18 | 18 | 37 | 0.0% | 0 | 0.997 | 0.809 | 0.561 | 0.4024 | 0.7393 | 0.3785 | PASS |
| 19_neurips_checklist | 493 | 33 | 13 | 26 | 3.0% | 0 | 0.998 | 0.787 | 0.7644 | 0.4671 | 0.8617 | 0.4585 | PASS |
| abstract | 131 | 8 | 14 | 33 | 0.0% | 0 | 0.997 | 0.853 | 0.7146 | 0.4325 | 0.6888 | 0.198 | PASS |

**Summary:** 13/20 sections pass

## Detailed Failures

### 01_introduction

**Failures:** metaphor_count (2 > 1), terminology_block_diffusion (3 > 2)

**Metaphors found:**
- "metered" in: "This is the decoding equivalent of a metered taxi: the fare
climbs with every token"
- "subway" in: "The cost is a subway: flat fare per station, no matter how many
tokens board"

**Long sentences (>40 words):**
- (41 words) "The three excluded, event\_type, affected\_entity and remediation\_hint, are the semantic fields whose gold labels are parser placeholders instead of ground truth, so scoring them measures agreement with Drain, not extraction quality; this was one of the nine corrections priced in Appendix~...."

**Terminology (block_diffusion):** 3 distinct names: DG, block-diffusion, diffusion model

### 02_cost_laws

**Failures:** metaphor_count (3 > 1), comma_splice_pct (15.2% > 10.0%)

**Metaphors found:**
- "taxi" in: "The taxi fare
climbs with every token; the subway charges per station"
- "metered" in: "AR is a metered taxi: the fare climbs with every token"
- "subway" in: "DG is a
subway: flat fare per station, no matter how many tokens board"

**Long sentences (>40 words):**
- (41 words) "The crossover X shifts with input length: DG's prompt is re-attended X times, making its input slope X2.6X steeper than the dense AR-3B's (0.123 against 0.047\,ms per prompt token), so longer prompts raise the base and push X upward ( Appendix~)...."
- (63 words) "Figure~ (page~) plots the two shapes against each other; Figure~ probes the staircase at its boundaries to show the steps are regular enough to be one constant. [FIGURE] [ht!] is a single constant (X, BGL, X). [/FIGURE] Fitting X to measurements at X and X splits the base into a X-independent floor ..."
- (44 words) "Two caveats bound this: HDFS is the 2,000-line LOGHUB sample~ over a full corpus, evaluated at X--X per cell against BGL's X; the AR-30B was re-measured on HDFS (p50 1082\,ms, CI [1078, 1106]) but the AR generalisation slope was fitted on the dense AR-3B...."

### 03_deployment

**Failures:** metaphor_count (2 > 1), number_density (4.2 > 4.0)

**Metaphors found:**
- "before you can blink" in: "Or, in human terms: DG finishes before you can blink; AR-30B
finishes before you finish reading this sentence"
- "knob" in: "The X knob is the difference: drop X from
8 to 4 and the model still fills every field, just with slightly
less precision, instead of cutting the JSON in half mid-key"

**Long sentences (>40 words):**
- (76 words) "Extending the same accounting past the table, the ratio keeps falling (X at X, X at X, where AR's plateau undercuts DG's accumulated canvas cost), so DG's serial advantage holds at typical output lengths but not universally. [TABLE] [h] evaluated at the canvas count each budget implies, anchored on ..."
- (52 words) "The two columns therefore compare AR at its natural output against DG at a full canvas budget, a conservative reading for DG, which emits four times more tokens at the last row. [/TABLE] The serial cost law gives crossover X tokens, DG is faster at any output length above X (integer-safe: X)...."

### 05_survives

**Failures:** metaphor_count (5 > 1), terminology_block_diffusion (3 > 2)

**Metaphors found:**
- "catching" in: "The gap widens in absolute terms
as hops accumulate, at eight hops, the difference between three
seconds and eight seconds is the difference between catching an
intruder in the act and finding the tra"
- "knob" in: "The X knob is DG's emergency brake: when a deadline tightens, drop
X from 8 to 4 and the model still fills every field, just with
slightly less precision, instead of cutting the JSON in half
mid-key"
- "knob" in: "The
X knob is a per-request quality slider: drop X to meet a tightened
deadline without truncating output"
- "taxi" in: "The taxi fare compounds; the subway fare doesn't"
- "knob" in: "[/FIGURE] 

The X knob need not be fixed"

**Long sentences (>40 words):**
- (60 words) "The five conditions that define the regime where block diffusion is the right choice are: [nosep] - Output length X tokens (DG crossover in serial). - Batch-1 by requirement: tenant isolation, interactivity, or serial chain structure. - Completeness is validated against a schema. - The quality floor..."
- (42 words) "Across schema widths from 12 to 100 fields, DG's latency advantage held between X and X (Figure~; measured against the dense AR-3B baseline, not AR-30B). [FIGURE] [ht!] 140\,ms too and the ratio stops growing. [/FIGURE] The X knob need not be fixed...."

**Terminology (block_diffusion):** 3 distinct names: DG, block-diffusion, diffusion model

### 07_threats

**Failures:** terminology_block_diffusion (3 > 2), passive_pct (33.9% > 25.0%)

**Terminology (block_diffusion):** 3 distinct names: DG, block-diffusion, diffusion model

### 08_conclusion

**Failures:** metaphor_count (2 > 1)

**Metaphors found:**
- "taxi" in: "The taxi fare climbs with every token; the subway charges per station"
- "knob" in: "The X knob is DG's emergency brake: when the deadline
tightens, DG degrades gracefully, lower quality, complete canvas"

### 16_appendix_forced

**Failures:** comma_splice_pct (14.3% > 10.0%)

**Long sentences (>40 words):**
- (57 words) "The advantage grows here, the opposite of Table~, because AR is no longer stopped by EOS at 126 tokens, so its line keeps climbing while DG adds one bounded canvas at a time. [/TABLE] So the figure quoted in the abstract is the conservative reading, not the optimistic one: it describes AR at its nat..."

## All Metaphors Found

- **01_introduction**: "metered" — "This is the decoding equivalent of a metered taxi: the fare
climbs with every token"
- **01_introduction**: "subway" — "The cost is a subway: flat fare per station, no matter how many
tokens board"
- **02_cost_laws**: "taxi" — "The taxi fare
climbs with every token; the subway charges per station"
- **02_cost_laws**: "metered" — "AR is a metered taxi: the fare climbs with every token"
- **02_cost_laws**: "subway" — "DG is a
subway: flat fare per station, no matter how many tokens board"
- **03_deployment**: "before you can blink" — "Or, in human terms: DG finishes before you can blink; AR-30B
finishes before you finish reading this sentence"
- **03_deployment**: "knob" — "The X knob is the difference: drop X from
8 to 4 and the model still fills every field, just with slightly
less precision, instead of cutting the JSON in half mid-key"
- **05_survives**: "catching" — "The gap widens in absolute terms
as hops accumulate, at eight hops, the difference between three
seconds and eight seconds is the difference between catching an
intruder in the act and finding the tra"
- **05_survives**: "knob" — "The X knob is DG's emergency brake: when a deadline tightens, drop
X from 8 to 4 and the model still fills every field, just with
slightly less precision, instead of cutting the JSON in half
mid-key"
- **05_survives**: "knob" — "The
X knob is a per-request quality slider: drop X to meet a tightened
deadline without truncating output"
- **05_survives**: "taxi" — "The taxi fare compounds; the subway fare doesn't"
- **05_survives**: "knob" — "[/FIGURE] 

The X knob need not be fixed"
- **08_conclusion**: "taxi" — "The taxi fare climbs with every token; the subway charges per station"
- **08_conclusion**: "knob" — "The X knob is DG's emergency brake: when the deadline
tightens, DG degrades gracefully, lower quality, complete canvas"
- **13_appendix_input_canvas**: "knob" — "A full X grid (canvas X, X, X per cell) confirms the single-knob finding
holds across all X values, not just X"
