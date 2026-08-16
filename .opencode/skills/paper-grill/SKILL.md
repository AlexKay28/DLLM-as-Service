---
name: paper-grill
description: Use when reviewing a research paper for validity, consistency, and clarity. Runs a systematic audit checking every number, claim, cross-reference, and narrative consistency across all sections. Triggers on phrases like "grill the paper", "review the paper", "check for contradictions", or "find issues".
---

# Paper Grill Skill

A systematic audit strategy for research papers that finds contradictions, unsupported claims, broken numbers, and narrative inconsistencies before a reviewer does.

## Audit Strategy

Run these 10 checks in order. For each, ask the question, search the paper, and flag any issue found. Fix critical issues before moving to the next check.

### Check 1: Number Consistency

**Question:** Does every number in the paper match its source data?

- Extract all numeric claims from each section (latencies, ratios, percentages, token counts)
- Compare against measured artifacts in `results/*.json`
- Compare against macros in `paper/assets/macros.tex`
- Verify calculated values (e.g., crossover = (dgBase - arBase) / arSlope)
- Check that tables, text, and figure captions all use the same numbers

**Commands:**
```bash
# Extract all numbers from macros
grep "newcommand" paper/assets/macros.tex
# Verify crossover calculation
python3 -c "print(f'L* = {(dgBase - arBase) / arSlope:.0f}')"
# Check table values match cost law
python3 -c "
ar_base, ar_slope = -5.8, 1.0
dg_base, dg_step, canvas = 378.8, 119.5, 256
for L in [123, 256, 512, 1024, 2048]:
    ar = ar_base + ar_slope * L
    nc = max(1, -(-L // canvas))
    dg = dg_base + dg_step * (nc - 1)
    print(f'L={L}: AR={ar:.0f} DG={dg:.0f} ratio={max(ar,dg)/min(ar,dg):.1f}x')
"
```

### Check 2: Claim vs Measurement

**Question:** Is every claim supported by actual measurement, or is it an extrapolation?

- Search for "forced", "projected", "fitted", "estimated" — these indicate non-measured values
- Verify: if Table says "measured", was it actually measured at that configuration?
- Check: are batch-throughput numbers being presented as serial p50 latency?
- Flag any mismatch between what was measured and what the paper claims

### Check 3: Cross-Reference Integrity

**Question:** Do all \ref{} and \label{} match? Are figures/tables referenced correctly?

```bash
# Find undefined references
cd paper && pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "undefined"
# Compare labels vs refs
echo "Labels:" && grep -h "\\\\label{" sections/*.tex | sort
echo "Refs:" && grep -oh "\\\\ref{[^}]*}" sections/*.tex main.tex | sort -u
```

### Check 4: Model Identity Consistency

**Question:** Does the paper use the same model name everywhere?

- Search for old model names that should have been replaced
- Check: if paper switched from dense AR-3B to MoE AR-30B, are there any orphaned "AR-3B" references?
- Verify figure legends match the text model names

```bash
grep -rn "AR-3B\|Qwen2.5-3B\|old_model_name" paper/sections/*.tex paper/main.tex
```

### Check 5: Framing Consistency

**Question:** Does the narrative tone match across sections?

- If abstract is positive ("advantage grows"), conclusion should not say "modest" or "thin"
- Check for contradictory adjectives: "real but thin", "narrow but real", "modest advantage"
- Ensure the story arc is consistent: setup → shapes → wins → limits → survives → conclusion

### Check 6: Equation Validity

**Question:** Are all equations physically meaningful?

- Check for negative intercepts — explain why (e.g., MoE efficient prefill)
- Verify variable definitions follow every equation ("where ...")
- Ensure general form appears before measured form
- Check that crossover formula uses the correct variables

### Check 7: Figure-Text Alignment

**Question:** Do figure captions match what the text claims?

- If Figure 1 caption says "4.3x at L=256" but text says "1.5x at L=256" — contradiction
- If figure uses data from a different model than the text describes — flag it
- Check that figure data sources are noted in captions

### Check 8: Appendix Consistency

**Question:** Do appendix numbers match the main text?

- Appendix tables should use the same cost law as the main text
- If Appendix H has a multi-record table, the AR values should match Table 2
- Check that appendix cross-references point to the right sections

### Check 9: Threats Completeness

**Question:** Does the threats section cover all known limitations?

- Energy: different idle baselines noted?
- Generalisation: measured on which model?
- Grammar-constrained decoding: attempted and reported?
- Cost law: fitted from batch or serial?
- Figure data: from which model?

### Check 10: Compilation Cleanliness

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

## Output Format

For each issue found, report:

```
Q[N]: [Question text]
ANSWER: [What was found]
STATUS: [CRITICAL / IMPORTANT / OK]
FIX: [What was done, or "no fix needed"]
```

After all 10 checks, summarize:
- Total issues found
- Critical issues fixed
- Remaining known limitations (acknowledged in paper)
- Compilation status
