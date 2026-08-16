---
name: paper-grill
description: Use when reviewing a research paper for validity, consistency, and clarity. Runs a systematic 15-check audit checking every number, claim, cross-reference, table consistency, figure alignment, equation validity, and narrative coherence. Triggers on phrases like "grill the paper", "review the paper", "check for contradictions", "find issues", or "audit paper".
---

# Paper Grill Skill

A systematic audit strategy for research papers that finds contradictions, unsupported claims, broken numbers, and narrative inconsistencies before a reviewer does. Run every check. For each, ask the question, search the paper, flag issues, and fix critical ones before moving on.

## Audit Strategy: 15 Checks

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

```bash
# Search for any old numbers
grep -rn "old_value\|6\.3\b\|1\.0.*ms.*token\|385\|59\b" paper/sections/*.tex paper/main.tex | grep -v "dense\|baseline\|earlier\|from 59\|0\.059\|0\.159"
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

## Output Format

For each check, report:

```
Q[N]: [Question text]
ANSWER: [What was found]
STATUS: [CRITICAL / IMPORTANT / OK]
FIX: [What was done, or "no fix needed"]
```

After all 15 checks, summarize:
- Total issues found
- Critical issues fixed
- Remaining known limitations (acknowledged in paper)
- Compilation status
- verify_numbers.py status
