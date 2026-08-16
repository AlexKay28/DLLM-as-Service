---
name: paper-grill
description: Use when reviewing a research paper for validity, consistency, and clarity. Runs a systematic 21-check audit checking every number, claim, cross-reference, table consistency, figure alignment, equation validity, narrative coherence, derived calculations, macro comments, crossover semantics, and per-event data attribution. Triggers on phrases like "grill the paper", "review the paper", "check for contradictions", "find issues", or "audit paper".
---

# Paper Grill Skill

A systematic audit strategy for research papers that finds contradictions, unsupported claims, broken numbers, and narrative inconsistencies before a reviewer does. Run every check. For each, ask the question, search the paper, flag issues, and fix critical ones before moving on.

## Audit Strategy: 21 Checks

### Check 1: Number Consistency — Macros vs Calculated vs Text

**Question:** Does every number in the paper match its source data?

Steps:
1. Read `paper/assets/macros.tex` — extract all numeric macros
2. Read measured artifacts in `results/*.json` — verify macro values
3. Recalculate derived values (crossover, ratios, energy) from first principles
4. Search every section file for hardcoded numbers — compare against macros
5. Verify table cells match both macros and calculations
6. Check that ratios (e.g., "15.6×") match the raw values they're derived from

```bash
# Extract macros
grep "newcommand" paper/assets/macros.tex
# Verify crossover
python3 -c "print(f'L* = {(dgBase - arBase) / arSlope:.1f}')"
# Check table values vs cost law
python3 -c "
ar_base, ar_slope = 79.1, 47.7
dg_base, dg_step, canvas = 378.8, 119.5, 256
for L in [32, 123, 256, 512, 1024]:
    ar = ar_base + ar_slope * L
    nc = max(1, -(-L // canvas))
    dg = dg_base + dg_step * (nc - 1)
    print(f'L={L}: AR={ar:.0f} DG={dg:.0f}')
"
```

### Check 2: Claim vs Measurement — Projected vs Measured

**Question:** Is every claim supported by actual measurement, or is it an extrapolation?

1. Search for "projected", "fitted", "extrapolated", "estimated", "upper bound"
2. Search for "measured", "serial", "batch-1", "batch throughput"
3. Verify: if Table says "measured", was it actually measured at that configuration?
4. Check: are batch-throughput numbers presented as serial p50 latency?
5. Verify: does the cost law use serial p50 or batch throughput (wall_time/n)?
6. Flag any mismatch between what was measured and what the paper claims

### Check 3: Cross-Reference Integrity

**Question:** Do all \ref{} and \label{} match? Are figures/tables referenced correctly?

```bash
cd paper && pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "undefined"
# Compare labels vs refs
echo "Labels:" && grep -h "\\\\label{" sections/*.tex | sort
echo "Refs:" && grep -oh "\\\\ref{[^}]*}" sections/*.tex main.tex | sort -u
# Check for orphaned labels (defined but never referenced)
# Check for dangling refs (referenced but never defined)
```

### Check 4: Model Identity Consistency

**Question:** Does the paper use the same model name everywhere?

1. Search for ALL model names (old and current) across all files
2. Check figure legends match text model names
3. Check macro comments match the model they claim to describe
4. Verify: if paper switched models, are there orphaned old references?
5. Check: do figures use data from the correct model?

```bash
grep -rn "AR-3B\|Qwen2.5-3B\|AR-30B\|Qwen3-30B\|AR-1.5B\|Qwen2.5-1.5B" paper/sections/*.tex paper/main.tex
# Check figure data sources
grep -n "p2_ar3b\|appD_ar3b\|issue111\|issue49" paper/figures/make_figures.py
```

### Check 5: Framing Consistency — Tone and Narrative Arc

**Question:** Does the narrative tone match across sections?

1. Read abstract, intro contributions, conclusion — do they tell the same story?
2. Check for contradictory adjectives: "real but thin", "narrow but real", "modest advantage"
3. Verify the story arc: setup → shapes → wins → limits → survives → conclusion
4. Check: if abstract says "advantage grows", conclusion should not say "narrows"
5. Check: if advantage NARROWS with output length (15.6× → 8.0×), text should say "narrows" not "grows"
6. Ensure positive framing where appropriate — don't undermine your own contribution

### Check 6: Equation Validity

**Question:** Are all equations physically meaningful and correctly defined?

1. Check for negative intercepts — explain WHY (e.g., MoE efficient prefill, fit artifact)
2. Verify variable definitions follow EVERY equation ("where ...")
3. Ensure general form appears before measured form
4. Check that crossover formula uses the correct variables
5. Verify R² values match the actual fit (not copied from a different model)
6. Check: does the equation's slope match the measured data's slope?
7. Verify: are serial vs batch slopes clearly distinguished?

### Check 7: Figure-Text Alignment

**Question:** Do figure captions match what the text claims?

1. Extract all ratios/numbers from figure captions
2. Extract corresponding claims from the text
3. Compare: if caption says "15.6× at L=256" but text says "8.0× at L=256" — contradiction
4. Check: if figure uses data from a different model than the text describes — flag it
5. Verify: figure data sources noted in captions (e.g., "AR data from dense baseline")
6. Check: do figure legends match the model names used in the text?
7. Verify: are figure ratios calculated from the same values as the text?

### Check 8: Appendix Consistency

**Question:** Do appendix numbers match the main text?

1. Compare appendix tables against main text tables (same values?)
2. Check: if Appendix H has a multi-record table, do the AR values match Table 2?
3. Verify: appendix cost law equations use the same macros/variables as main text
4. Check: do appendix cross-references point to the right sections?
5. Verify: appendix R² values match the actual data (not stale from old model)
6. Check: are appendix values computed from the correct model's data?

### Check 9: Table Completeness

**Question:** Are all relevant models in every table?

1. Check: does every table include ALL models mentioned in the paper?
2. If paper compares AR-30B, AR-1.5B, and DG — do all three appear in every comparison table?
3. Check: are table columns labeled correctly (serial vs batch, ms vs ms/token)?
4. Verify: do table values match the cost law at those output lengths?
5. Check: is there a "max_tokens" vs "actual output" distinction where needed?
6. Verify: are events/hr calculations correct (3600 / p50_seconds)?

### Check 10: Threats Completeness

**Question:** Does the threats section cover all known limitations?

1. Energy: different idle baselines noted?
2. Generalisation: measured on which model?
3. Grammar-constrained decoding: attempted and reported?
4. Cost law: fitted from batch or serial? (must match what text claims)
5. Figure data: from which model?
6. Quality comparison: which AR model was used for F1?
7. Batch vs serial: is the gap disclosed?
8. Engine version: was AR-30B measured on the same vLLM version as DG?

### Check 11: Stale References — Old Numbers After Model Switch

**Question:** After switching models (e.g., dense AR-3B → MoE AR-30B), are there leftover old numbers?

1. Search for old slope values (e.g., "6.3" or "1.0" if current is "47.7")
2. Search for old crossover values (e.g., "59" or "385" if current is "6")
3. Search for old advantage ratios (e.g., "1.4×" or "5.4×" if current is "15.6×")
4. Search for old intercept values (e.g., "-5.8" or "7.3" if current is "79.1")
5. Search for old batch multiplier values (e.g., "42.8" if current is "77")
6. Check ALL sections — intro, conclusion, appendices, captions, contrast tables
7. **Verify batch-throughput slopes:** if the text says "batch-throughput slope (1.0 ms/token)", check if 1.0 is derivable from the batch sweep JSON. Compute `batch300_per_event_ms / mean_output_tokens` and compare. If the text says "48× steeper" but the data gives 73×, the slope and ratio are both stale.
8. **Check for orphaned macros files:** if `main.tex` loads `../assets/macros.tex`, verify no stale `paper/v4/macros.tex` exists with conflicting values.

```bash
# Search for any old numbers
grep -rn "old_value\|6\.3\b\|1\.0.*ms.*token\|385\|59\b" paper/sections/*.tex paper/main.tex | grep -v "dense\|baseline\|earlier\|from 59\|0\.059\|0\.159"
# Verify batch-throughput slope
python3 -c "
import json
d = json.load(open('results/issue111_qwen3_batch_sweep.json'))
print(f'Batch per-token: {d[\"batch300_per_event_ms\"]}/123 = {d[\"batch300_per_event_ms\"]/123:.2f} ms/token')
print(f'Serial/batch ratio: 47.7/{d[\"batch300_per_event_ms\"]/123:.2f} = {47.7/(d[\"batch300_per_event_ms\"]/123):.0f}x')
"
# Check for orphaned macros files
ls paper/v4/macros.tex 2>/dev/null && echo 'WARNING: orphaned macros.tex exists' || echo 'OK: no orphan'
```

### Check 12: Cost-Per-Event Table Consistency

**Question:** Are all cost calculations correct and complete?

1. Verify: events/hr = 3600 / (p50_ms / 1000)
2. Verify: $/event = hourly_rate / events_hr
3. Check: are ALL models in the table (AR-30B, AR-1.5B, DG)?
4. Check: is the cost ratio R_cost calculated from the right pair?
5. Verify: does the co-residency argument use the right model (smallest AR)?
6. Check: are duty cycle calculations correct (25% = events_hr × 0.25)?

### Check 13: Contrast Table Alignment

**Question:** Does the practical challenge contrast table match the paper's claims?

1. Check: does "crossover at L=X" match the actual crossover value?
2. Check: does "slope = X ms/token" match the cost law?
3. Check: does "batch multiplier = X" match the measured value?
4. Check: does "tail CV = X%" match the measured value?
5. Verify: are the "key driver" column entries accurate?

### Check 14: Conclusion vs Abstract vs Contributions

**Question:** Do the three summary locations tell the same story?

1. Extract all numbers from abstract
2. Extract all numbers from contributions block (intro)
3. Extract all numbers from conclusion
4. Compare: do they all cite the same advantage ratios?
5. Compare: do they all cite the same crossover value?
6. Compare: do they all cite the same batch multiplier?
7. Check: is the framing consistent (all positive, or all neutral)?

### Check 15: Compilation Cleanliness

**Question:** Does the paper compile without errors?

```bash
cd paper && pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -5
bibtex main 2>&1 | tail -2
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -3
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -3
echo "errors: $(grep -c 'Error' main.log)"
echo "overfull: $(grep -c 'Overfull' main.log)"
echo "undefined: $(grep -c 'undefined' main.log)"
```

Also run:
```bash
python3 paper/verify_numbers.py
```

### Check 16: Derived Calculation Verification

**Question:** Do all derived values (ratios, percentages, crossovers) actually compute to their stated values from their raw inputs?

This check catches cases where a ratio macro (e.g., `energyRatio = 2.4`) doesn't match the division of its inputs (e.g., `arThreeEnergy / dgEnergy = 10.1 / 4.2`). The inputs may be correct individually, but the derived value may be stale from an older measurement.

1. List every derived macro (ratios, percentages, crossovers) and identify its formula
2. Recompute each from its stated inputs
3. Flag any mismatch — the derived value is likely stale
4. Check rounding-order issues: if `dgMarginal = 0.47` (rounded from `119.5/256 = 0.4668`), then `marginalRatio = arSlope / dgMarginal` depends on whether you use the rounded or unrounded value
5. Verify cost-per-event table arithmetic: `events/hr = 3600000 / p50_ms`, `$/event = gpu_rate / events_hr`

```bash
python3 -c "
# Recompute every derived macro
checks = [
    ('crossoverL', (378.8 - 79.1) / 47.7, 6),
    ('energyRatio', 10.1 / 4.2, 2.4),
    ('footprintRatio', 52.0 / 3.0, 17),
    ('inputSlopeRatio', 0.1 / 0.038, 2.6),
    ('marginalRatio', 47.7 / 0.47, 101.5),  # uses rounded dgMarginal
    ('crossoverLambda', 1000 / 378.8, 2.6),
    ('arOneFiveEventsHr', 3600000 / 665.7, 5408),
    ('defectTwoRatio', 1841.6 / 413.3, 4.5),
]
for name, computed, macro_val in checks:
    match = '✓' if abs(computed - macro_val) < 0.5 else '✗ MISMATCH'
    print(f'{name}: computed={computed:.2f}, macro={macro_val} {match}')
"
```

### Check 17: Macro Comment Accuracy

**Question:** Do macro comments correctly identify the source model and artifact?

A macro's VALUE may be correct, but its COMMENT may say "AR-30B" when the data is actually from the dense AR-3B. This is a maintenance hazard — a future editor trusts the comment, not the JSON.

1. For every macro with a model name in its comment (e.g., "AR-30B", "AR-3B", "DG"), verify the source JSON's `provenance.model_name` matches
2. Check: if the comment says "AR-30B" but the value is ~1616ms (AR-3B's p50), the comment is wrong
3. Check: if the comment says "AR-30B" but the source file is `issue43/ar3b_L256.json`, the comment is wrong
4. Verify: do `arHdfs`/`arBgl` comments match the actual model? (values ~1616ms = AR-3B, not AR-30B at ~5900ms)

```bash
# Cross-check macro comments against source file provenance
grep "AR-30B\|AR-3B\|dense" paper/assets/macros.tex | while read line; do
    macro=$(echo "$line" | grep -oP '\\\w+(?=\{)')
    comment_model=$(echo "$line" | grep -oP 'AR-\d+B')
    echo "Macro: $macro, Comment says: $comment_model"
done
# Then verify each against its source JSON's provenance.model_name
```

### Check 18: Crossover Semantics — Integer Safety

**Question:** Are "above X" and "≥ X" claims integer-safe?

If the crossover L* = 6.28 (rounded to 6 in the macro), then:
- "above 6 tokens" is ambiguous: at L=6, AR is still faster (365ms vs 379ms)
- "≥ 6 tokens" is WRONG: at L=6, AR is faster
- "≥ 7 tokens" or "above L* ≈ 6" is correct

1. Compute the TRUE crossover (not the rounded macro value)
2. Check the crossover value in the text: does "above X" include the crossover point itself?
3. Check condition lists (e.g., "five conditions"): does "≥ X" include a point where AR is still faster?
4. Fix: use "≥ ⌈L*⌉" (ceiling) or "above L*" or "≥ 7 tokens" (integer-safe)

```bash
python3 -c "
ar_base, ar_slope, dg_base = 79.1, 47.7, 378.8
crossover = (dg_base - ar_base) / ar_slope
print(f'True crossover: {crossover:.4f}')
print(f'At L={int(crossover)}: AR={ar_base+ar_slope*int(crossover):.1f}, DG={dg_base} -> {\"AR faster\" if ar_base+ar_slope*int(crossover) < dg_base else \"DG faster\"}')
print(f'At L={int(crossover)+1}: AR={ar_base+ar_slope*(int(crossover)+1):.1f}, DG={dg_base} -> {\"AR faster\" if ar_base+ar_slope*(int(crossover)+1) < dg_base else \"DG faster\"}')
print(f'Integer-safe: >= {int(crossover)+1}')
"
# Check all "above N" and ">= N" claims in the paper
grep -n 'above.*token\|≥.*token\|geq.*token' paper/v4/sections/*.tex
```

### Check 19: Per-Event Data Attribution

**Question:** Do per-event prediction files (`.jsonl`) correspond to the model named in the text?

A paper may label per-event statistics (CV, max/median) as "AR-30B" when the actual `.jsonl` file contains data from the dense AR-3B. The median latency in the file reveals the true model.

1. For every per-event statistic cited in the paper, identify the source `.jsonl` file
2. Compute the median latency from that file
3. Compare: does the median match the named model's p50? (AR-3B ~1616ms, AR-30B ~5900ms)
4. Check: does the file naming match the model? (`ar3b_L256_r1_preds.jsonl` = AR-3B, not AR-30B)
5. Flag any mismatch between the text's model label and the data file's actual model

```bash
python3 -c "
import json, statistics
for f in ['results/anchors/ar3b_L256_r1_preds.jsonl', 'results/anchors/dg_L256_r1_preds.jsonl']:
    lats = [json.loads(l)['latency_ms'] for l in open(f) if l.strip()]
    print(f'{f}: n={len(lats)}, median={statistics.median(lats):.0f}ms')
    # AR-3B p50 ~1616ms, AR-30B p50 ~5900ms, DG p50 ~379ms
"
# Check: are there AR-30B per-event files? If not, per-event stats can't be AR-30B
ls results/anchors/ar*preds* results/issue111*preds* 2>/dev/null || echo 'No AR-30B per-event files'
```

### Check 20: Verify Script Reality Check

**Question:** Does `verify_numbers.py` actually do what `macros.tex` claims it does?

The `macros.tex` header says "verify_numbers.py re-reads those artifacts." If the script only checks provenance blocks and decimal-place policy (not actual values), the claim is false and value mismatches will go undetected.

1. Read the `macros.tex` header comment — what does it claim `verify_numbers.py` does?
2. Read `verify_numbers.py` — what does it actually check?
3. If the script checks precision but not values, any value mismatch (e.g., macro says 385.5, JSON says 368.5) will pass
4. Flag: the header comment overstates the script's capability
5. Consider: add a value-verification pass to the script, or fix the comment

```bash
# Check what verify_numbers.py actually does
grep -n 'def check\|json.load\|value.*macro\|re-read' paper/verify_numbers.py
# Run it and see if it catches a deliberate error
python3 paper/verify_numbers.py 2>&1 | tail -5
```

### Check 21: Dead Macro Detection

**Question:** Are there macros defined but never used in any section file?

Dead macros are maintenance hazards — they may contain stale values from previous versions and mislead anyone who reads `macros.tex` expecting all defined macros to be live.

1. Extract all macro definitions from `macros.tex`
2. Search all `.tex` files (sections + main) for each macro name
3. Flag any macro that is defined but never referenced
4. Check: are there model-specific macros (e.g., `arOneFiveTwoFive`) for a model that's no longer discussed?
5. Consider: either remove dead macros or add a comment explaining why they're retained

```bash
python3 -c "
import re, os
defined = set(re.findall(r'\\\\newcommand\{\\\\(\w+)\}', open('paper/assets/macros.tex').read()))
used = set()
for root, dirs, files in os.walk('paper/v4'):
    for f in files:
        if f.endswith('.tex'):
            used.update(re.findall(r'\\\\(\w+)', open(os.path.join(root, f)).read()))
dead = defined - used
print(f'Defined: {len(defined)}, Used: {len(defined & used)}, Dead: {len(dead)}')
for m in sorted(dead):
    print(f'  DEAD: \\\\{m}')
"
```

## Output Format

For each check, report:

```
Q[N]: [Question text]
ANSWER: [What was found]
STATUS: [CRITICAL / IMPORTANT / OK]
FIX: [What was done, or "no fix needed"]
```

After all 21 checks, summarize:
- Total issues found
- Critical issues fixed
- Remaining known limitations (acknowledged in paper)
- Compilation status
- verify_numbers.py status
- Dead macros count
- Derived calculation mismatches count
