---
name: paper-grill
description: Use when reviewing a research paper for validity, consistency, and clarity. Runs a systematic 24-check audit across 6 dimensions — data integrity, narrative quality, mathematical correctness, presentation, completeness, and maintenance hygiene. Triggers on phrases like "grill the paper", "review the paper", "check for contradictions", "find issues", or "audit paper".
---

# Paper Grill Skill

A systematic audit strategy for research papers that finds contradictions, unsupported claims, broken numbers, narrative inconsistencies, and readability barriers before a reviewer does. Run every check. For each, ask the question, search the paper, flag issues, and fix critical ones before moving on.

## Workflow

1. **Read** `macros.tex`, all section files, and the `results/*.json` artifacts in parallel
2. **Run Group A** (data integrity) first — fix any CRITICAL number mismatches before continuing
3. **Run Group B** (narrative) — assess storytelling, readability, definitions
4. **Run Group C** (math) — verify equations and derived calculations
5. **Run Groups D–F** (presentation, completeness, hygiene) — can run in parallel
6. **Compile** after all fixes
7. **Report** using the output format at the end

## Audit Strategy: 24 Checks in 6 Groups

---

### Group A: Data Integrity

#### Check 1: Number Consistency — Macros vs Calculated vs Text

**Question:** Does every number in the paper match its source data?

Steps:
1. Read `paper/assets/macros.tex` — extract all numeric macros
2. Read measured artifacts in `results/*.json` — verify macro values match JSON fields
3. Recalculate derived values (crossover, ratios, energy) from first principles
4. Search every section file for hardcoded numbers — compare against macros
5. Verify table cells match both macros and calculations
6. Check that ratios (e.g., "15.6×") match the raw values they're derived from

```bash
# Extract macros and their source comments
grep "newcommand" paper/assets/macros.tex
# Verify crossover calculation (use CURRENT macro values, not stale ones)
python3 -c "
ar_base, ar_slope, dg_base = 79.1, 47.7, 378.8
print(f'L* = ({dg_base} - {ar_base}) / {ar_slope} = {(dg_base - ar_base) / ar_slope:.2f}')
"
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

#### Check 2: Claim vs Measurement — Projected vs Measured

**Question:** Is every claim supported by actual measurement, or is it an extrapolation?

1. Search for "projected", "fitted", "extrapolated", "estimated", "upper bound"
2. Search for "measured", "serial", "batch-1", "batch throughput"
3. Verify: if Table says "measured", was it actually measured at that configuration?
4. Check: are batch-throughput numbers presented as serial p50 latency?
5. Verify: does the cost law use serial p50 or batch throughput (wall_time/n)?
6. Flag any mismatch between what was measured and what the paper claims

#### Check 3: Cross-Reference Integrity

**Question:** Do all \ref{} and \label{} match? Are figures/tables referenced correctly?

```bash
cd paper && pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "undefined"
# Compare labels vs refs
echo "Labels:" && grep -h "\\\\label{" sections/*.tex | sort
echo "Refs:" && grep -oh "\\\\ref{[^}]*}" sections/*.tex main.tex | sort -u
```

#### Check 4: Model Identity Consistency

**Question:** Does the paper use the same model name everywhere? Are macro comments accurate?

1. Search for ALL model names (old and current) across all files
2. Check figure legends match text model names
3. Check macro COMMENTS match the model the data actually came from (value tells the truth: ~1616ms = AR-3B, ~5900ms = AR-30B)
4. Verify: if paper switched models, are there orphaned old references?
5. Check: do figures use data from the correct model?

```bash
grep -rn "AR-3B\|Qwen2.5-3B\|AR-30B\|Qwen3-30B\|AR-1.5B\|Qwen2.5-1.5B" paper/v4/sections/*.tex
# Cross-check macro comments: if comment says "AR-30B" but value is ~1616, it's wrong
grep "AR-30B\|AR-3B" paper/assets/macros.tex
```

---

### Group B: Narrative Quality

#### Check 5: Framing Consistency — Tone and Narrative Arc

**Question:** Does the narrative tone match across sections?

1. Read abstract, intro contributions, conclusion — do they tell the same story?
2. Check for contradictory adjectives: "real but thin", "narrow but real", "modest advantage"
3. Verify the story arc: setup → shapes → wins → limits → survives → conclusion
4. Check: if abstract says "advantage grows", conclusion should not say "narrows"
5. Check: if advantage NARROWS with output length (15.6× → 8.0×), text should say "narrows" not "grows"

#### Check 22: Storytelling and Narrative Coherence

**Question:** Does the paper tell a single, compelling, logically-ordered story — or does it read as a list of results stitched together?

1. **Story arc integrity:** Read the first and last sentence of each section. Does each section's opening follow from the previous section's closing? Write a one-sentence summary of each section; the list should read as a coherent paragraph.
2. **Setup-payoff closure:** For every setup (e.g., "we derive two cost laws" in the intro), verify the payoff appears (the laws in §2, the measurement in §3). For every payoff (e.g., "77× batching gap"), verify the setup was laid earlier. Flag orphaned setups (promised but never delivered) and orphaned payoffs (delivered but never promised).
3. **Contribution-to-section mapping:** List the paper's stated contributions (usually in the intro). For each contribution, identify which section delivers it. Flag any contribution with no corresponding section, or any section that delivers nothing promised in the intro.
4. **Forward references and foreshadowing:** Check phrases like "see Appendix X", "as shown later", "we return to this in §Y". Does the referenced section actually address the foreshadowed point? Flag broken foreshadowing.
5. **Tension and resolution:** Does the paper create genuine tension (a problem the reader wants solved) and resolve it? Or does it present results without framing why they matter? Flag sections that present data without first stating the question the data answers.
6. **Narrative consistency across passes:** If the paper has been revised (model switches, number corrections), check that the narrative was updated alongside the numbers.

```bash
# Extract first and last sentence of each section
python3 -c "
import re, os
for f in sorted(os.listdir('paper/v4/sections')):
    if f.endswith('.tex'):
        text = open(f'paper/v4/sections/{f}').read()
        clean = re.sub(r'\\\\\w+(\{[^}]*\})?', '', text)
        sentences = re.split(r'(?<=[.!?])\s+', clean.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if sentences:
            print(f'=== {f} ===')
            print(f'  OPEN: {sentences[0][:120]}')
            print(f'  CLOSE: {sentences[-1][:120]}')
            print()
"
# Check contribution-to-section mapping
grep -n 'textbf{.*}\|\\\\item' paper/v4/sections/01_introduction.tex | head -20
```

#### Check 23: Language and Readability Hardness

**Question:** Is the prose clear and readable, or is it needlessly dense, jargon-stacked, and clause-nested?

1. **Sentence length distribution:** Compute the word count of every sentence. Flag sentences over 40 words (hard to parse) and over 60 words (likely unreadable). The median should be 15–25 words. **Note:** equations + where-blocks may be parsed as single "sentences" by regex — inspect flagged sentences manually to filter false positives.
2. **Nested clause depth:** Count subordinate clauses (indicated by "which", "where", "that", em-dashes, parentheticals). More than 3 levels of nesting makes a sentence hard to follow.
3. **Jargon density per sentence:** Count domain-specific terms per sentence. A sentence with 5+ jargon terms and no plain-language restatement is a wall.
4. **Acronym/abbreviation expansion:** Every acronym must be expanded on first use. Flag bare acronyms that appear before their expansion.
5. **Passive voice overuse:** Flag paragraphs with 4+ consecutive passive-voice sentences.
6. **Hedging language:** "may", "might", "could", "possibly" weaken claims. Flag sentences with 2+ hedges.
7. **Readability formula:** Compute the Flesch Reading Ease score per section. Target: 30–50. Below 20 is unreadable; above 60 may be too informal.
8. **Paragraph opening strength:** Check for buried leads ("It should be noted that...").

```bash
python3 -c "
import re, os
try:
    import textstat
except ImportError:
    print('Install: pip install textstat'); exit()
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    clean = re.sub(r'\\\\\w+(\[[^\]]*\])?(\{[^}]*\})?', '', text)
    clean = re.sub(r'%.*', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    sentences = [s for s in sentences if len(s.split()) > 3]
    word_counts = [len(s.split()) for s in sentences]
    long = [c for c in word_counts if c > 40]
    very_long = [c for c in word_counts if c > 60]
    fre = textstat.flesch_reading_ease(clean)
    print(f'{f}: median={sorted(word_counts)[len(word_counts)//2] if word_counts else 0}w  >40w:{len(long)}  >60w:{len(very_long)}  FRE={fre:.1f}')
"
# Check acronym expansion on first use
grep -n 'was measured\|is shown\|was found\|was used\|were used' paper/v4/sections/*.tex
grep -n 'may be\|might be\|could be\|possibly\|it appears\|it seems' paper/v4/sections/*.tex
```

#### Check 24: Information Completeness and Definitions

**Question:** Does the paper define every term, variable, and concept before using it — and does it omit information a reviewer needs?

1. **Variable definition completeness:** For every equation, check that every variable is defined in the surrounding "where..." block. **Use a 1000-char search window** (not 500) after each equation to find the where-block; shorter windows cause false positives when the where-block follows a display break.
2. **First-use definitions:** For every technical term (e.g., "canvas", "denoising step", "prefix caching", "first-object-wins parser", "p99", "max/median ratio"), verify it is defined or explained on first use.
3. **Methodology gaps:** For every experiment cited, check: (a) what model was used, (b) what n was, (c) what hardware/software stack, (d) what metric was measured, (e) how many repeats.
4. **Missing baselines:** Does the paper compare against all relevant alternatives?
5. **Missing error bars / significance:** For every numeric claim, check: is a CI, SE, or CV reported? If a difference is claimed, is a statistical test cited?
6. **Missing scale information:** Are model sizes, GPU, vLLM/torch version stated for every result?
7. **Ambiguous antecedents:** Check "this approach", "that model", "the baseline" — does the referent unambiguously resolve?
8. **Undefined metrics:** For every metric (F1, schema-validity, consistency, anomaly-F1, CV, max/median), check: is the metric defined before the first numeric value is reported?
9. **Missing experiment parameters:** For cited results, check: what was K? Canvas size? max_tokens? Prompt length?
10. **Forward-reference dependency chains:** If §3 says "see Appendix X" and Appendix X says "as described in §3", the reader is in a loop.

```bash
# Check variable definitions after equations (1000-char window)
python3 -c "
import re, os
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    for m in re.finditer(r'\\\\begin\{(equation|align)\*?\}.*?\\\\end\{\1\*?\}', text, re.DOTALL):
        eq = m.group(0)
        after = text[m.end():m.end()+1000]  # 1000-char window, not 500
        has_where = 'where' in after.lower() or 'in which' in after.lower()
        if not has_where:
            print(f'{f}: equation without where-block (within 1000 chars)')
"
# Check for undefined terms before first definition
grep -n 'canvas\|denoising\|prefix caching\|first-object-wins\|p99\|max/median\|KV cache\|MoE' paper/v4/sections/01_introduction.tex
# Check ambiguous 'this'/'that' references
grep -n 'this approach\|that model\|the baseline\|this method\|that result' paper/v4/sections/*.tex
```

---

### Group C: Mathematical Correctness

#### Check 6: Equation Validity

**Question:** Are all equations physically meaningful and correctly defined?

1. Check for negative intercepts — explain WHY (e.g., MoE efficient prefill, fit artifact)
2. Verify variable definitions follow EVERY equation ("where ...")
3. Ensure general form appears before measured form
4. Check that crossover formula uses the correct variables
5. Verify R² values match the actual fit
6. Check: does the equation's slope match the measured data's slope?
7. Verify: are serial vs batch slopes clearly distinguished?

#### Check 16: Derived Calculation Verification

**Question:** Do all derived values (ratios, percentages, crossovers) actually compute to their stated values from their raw inputs?

This catches cases where a ratio macro doesn't match the division of its inputs. The inputs may be correct individually, but the derived value may be stale from an older measurement.

1. List every derived macro (ratios, percentages, crossovers) and identify its formula
2. Recompute each from its stated inputs
3. Flag any mismatch — the derived value is likely stale
4. Check rounding-order issues: if `dgMarginal = 0.47` (rounded from `119.5/256 = 0.4668`), then `marginalRatio = arSlope / dgMarginal` depends on whether you use the rounded or unrounded value
5. Verify cost-per-event table arithmetic: `events/hr = 3600000 / p50_ms`, `$/event = gpu_rate / events_hr`

```bash
python3 -c "
checks = [
    ('crossoverL', (378.8 - 79.1) / 47.7, 6),
    ('energyRatio', 10.1 / 4.2, 2.4),
    ('footprintRatio', 52.0 / 3.0, 17),
    ('inputSlopeRatio', 0.1 / 0.038, 2.6),
    ('marginalRatio', 47.7 / 0.47, 101.5),
    ('crossoverLambda', 1000 / 378.8, 2.6),
    ('arOneFiveEventsHr', 3600000 / 665.7, 5408),
    ('defectTwoRatio', 1841.6 / 413.3, 4.5),
]
for name, computed, macro_val in checks:
    match = 'OK' if abs(computed - macro_val) < 0.5 else 'MISMATCH'
    print(f'{name}: computed={computed:.2f}, macro={macro_val} {match}')
"
```

#### Check 18: Crossover Semantics — Integer Safety

**Question:** Are "above X" and "≥ X" claims integer-safe?

If the crossover L* = 6.28 (rounded to 6 in the macro), then:
- "above 6 tokens" is ambiguous: at L=6, AR is still faster
- "≥ 6 tokens" is WRONG: at L=6, AR is faster
- "≥ 7 tokens" or "above L* ≈ 6" is correct

1. Compute the TRUE crossover (not the rounded macro value)
2. Check the crossover value in the text: does "above X" include the crossover point itself?
3. Check condition lists: does "≥ X" include a point where AR is still faster?
4. Fix: use "≥ ⌈L*⌉" (ceiling) or "above L*" or "≥ 7 tokens" (integer-safe)

```bash
python3 -c "
ar_base, ar_slope, dg_base = 79.1, 47.7, 378.8
crossover = (dg_base - ar_base) / ar_slope
print(f'True crossover: {crossover:.4f}')
print(f'At L={int(crossover)}: AR={ar_base+ar_slope*int(crossover):.1f}, DG={dg_base}')
print(f'At L={int(crossover)+1}: AR={ar_base+ar_slope*(int(crossover)+1):.1f}, DG={dg_base}')
print(f'Integer-safe: >= {int(crossover)+1}')
"
grep -n 'above.*token\|≥.*token\|geq.*token' paper/v4/sections/*.tex
```

---

### Group D: Presentation

#### Check 7: Figure-Text Alignment

**Question:** Do figure captions match what the text claims?

1. Extract all ratios/numbers from figure captions
2. Extract corresponding claims from the text
3. Compare: if caption says "15.6× at L=256" but text says "8.0× at L=256" — contradiction
4. Check: if figure uses data from a different model than the text describes — flag it
5. Verify: figure data sources noted in captions
6. Check: do figure legends match the model names used in the text?

#### Check 8: Appendix Consistency

**Question:** Do appendix numbers match the main text?

1. Compare appendix tables against main text tables (same values?)
2. Check: if Appendix H has a multi-record table, do the AR values match Table 2?
3. Verify: appendix cost law equations use the same macros/variables as main text
4. Check: do appendix cross-references point to the right sections?
5. Verify: appendix R² values match the actual data
6. Check: are appendix values computed from the correct model's data?

#### Check 9: Table Completeness

**Question:** Are all relevant models in every table?

1. Check: does every table include ALL models mentioned in the paper?
2. Check: are table columns labeled correctly (serial vs batch, ms vs ms/token)?
3. Verify: do table values match the cost law at those output lengths?
4. Check: is there a "max_tokens" vs "actual output" distinction where needed?
5. Verify: are events/hr calculations correct (3600000 / p50_ms)?

#### Check 13: Contrast Table Alignment

**Question:** Does the practical challenge contrast table match the paper's claims?

1. Check: does "crossover at L=X" match the actual crossover value?
2. Check: does "slope = X ms/token" match the cost law?
3. Check: does "batch multiplier = X" match the measured value?
4. Check: does "tail CV = X%" match the measured value?
5. Verify: are the "key driver" column entries accurate?
6. **Check: does any cell mix data from different models without disclosure?** (e.g., AR-3B tail stats + AR-30B energy in one cell labeled "AR-30B")

---

### Group E: Completeness

#### Check 10: Threats Completeness

**Question:** Does the threats section cover all known limitations?

1. Energy: different idle baselines noted?
2. Generalisation: measured on which model?
3. Grammar-constrained decoding: attempted and reported?
4. Cost law: fitted from batch or serial? (must match what text claims)
5. Figure data: from which model?
6. Quality comparison: which AR model was used for F1?
7. Batch vs serial: is the gap disclosed?
8. Engine version: was AR-30B measured on the same vLLM version as DG?
9. **Per-event data attribution:** if per-event CV/max-median is labeled "AR-30B", was it actually measured on AR-30B? (check the `.jsonl` source file — the median latency reveals the true model)
10. **Derived value staleness:** if a derived macro doesn't match its inputs, is this noted?

#### Check 14: Conclusion vs Abstract vs Contributions

**Question:** Do the three summary locations tell the same story?

1. Extract all numbers from abstract
2. Extract all numbers from contributions block (intro)
3. Extract all numbers from conclusion
4. Compare: do they all cite the same advantage ratios?
5. Compare: do they all cite the same crossover value?
6. Compare: do they all cite the same batch multiplier?
7. Check: is the framing consistent (all positive, or all neutral)?

---

### Group F: Maintenance Hygiene

#### Check 11: Stale References — Old Numbers After Model Switch

**Question:** After switching models, are there leftover old numbers?

1. Search for old slope values (e.g., "6.3" or "1.0" if current is "47.7")
2. Search for old crossover values (e.g., "59" or "385" if current is "6")
3. Search for old advantage ratios (e.g., "1.4×" if current is "15.6×")
4. Search for old intercept values (e.g., "-5.8" if current is "79.1")
5. Search for old batch multiplier values
6. Check ALL sections — intro, conclusion, appendices, captions, contrast tables
7. **Verify batch-throughput slopes:** if the text says "batch-throughput slope (1.0 ms/token)", check if 1.0 is derivable from the batch sweep JSON. Compute `batch300_per_event_ms / mean_output_tokens` and compare.
8. **Check for orphaned macros files:** if `main.tex` loads `../assets/macros.tex`, verify no stale `paper/v4/macros.tex` exists with conflicting values.

```bash
# Verify batch-throughput slope
python3 -c "
import json
d = json.load(open('results/issue111_qwen3_batch_sweep.json'))
per_token = d['batch300_per_event_ms'] / 123
print(f'Batch per-token: {d[\"batch300_per_event_ms\"]}/123 = {per_token:.2f} ms/token')
print(f'Serial/batch ratio: 47.7/{per_token:.2f} = {47.7/per_token:.0f}x')
"
# Check for orphaned macros files
ls paper/v4/macros.tex 2>/dev/null && echo 'WARNING: orphaned macros.tex' || echo 'OK: no orphan'
```

#### Check 15: Compilation Cleanliness

**Question:** Does the paper compile without errors?

```bash
cd paper/v4 && pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -3
bibtex main 2>&1 | tail -2
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -3
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -3
echo "errors: $(grep -c 'Error' main.log)"
echo "overfull: $(grep -c 'Overfull' main.log)"
echo "undefined: $(grep -c 'undefined' main.log)"
```

#### Check 17: Macro Comment Accuracy

**Question:** Do macro comments correctly identify the source model and artifact?

A macro's VALUE may be correct, but its COMMENT may say "AR-30B" when the data is actually from the dense AR-3B.

1. For every macro with a model name in its comment, verify the source JSON's `provenance.model_name` matches
2. Check: if the comment says "AR-30B" but the value is ~1616ms (AR-3B's p50), the comment is wrong
3. Check: if the comment says "AR-30B" but the source file is `issue43/ar3b_L256.json`, the comment is wrong

#### Check 19: Per-Event Data Attribution

**Question:** Do per-event prediction files (`.jsonl`) correspond to the model named in the text?

1. For every per-event statistic cited, identify the source `.jsonl` file
2. Compute the median latency from that file
3. Compare: does the median match the named model's p50? (AR-3B ~1616ms, AR-30B ~5900ms)
4. Check: does the file naming match the model?
5. Flag any mismatch between the text's model label and the data file's actual model

```bash
python3 -c "
import json, statistics
for f in ['results/anchors/ar3b_L256_r1_preds.jsonl', 'results/anchors/dg_L256_r1_preds.jsonl']:
    lats = [json.loads(l)['latency_ms'] for l in open(f) if l.strip()]
    print(f'{f}: n={len(lats)}, median={statistics.median(lats):.0f}ms')
"
ls results/anchors/ar*preds* results/issue111*preds* 2>/dev/null || echo 'No AR-30B per-event files'
```

#### Check 20: Verify Script Reality Check

**Question:** Does `verify_numbers.py` actually do what `macros.tex` claims it does?

1. Read the `macros.tex` header comment — what does it claim?
2. Read `verify_numbers.py` — what does it actually check?
3. If the script checks precision but not values, any value mismatch will pass undetected
4. Flag: the header comment overstates the script's capability
5. Fix: either add a value-verification pass to the script, or correct the comment

```bash
grep -n 'def check\|json.load\|value.*macro\|re-read' paper/verify_numbers.py
python3 paper/verify_numbers.py 2>&1 | tail -5
```

#### Check 21: Dead Macro Detection

**Question:** Are there macros defined but never used in any section file?

1. Extract all macro definitions from `macros.tex`
2. Search all `.tex` files (sections + main) for each macro name — **exclude `macros.tex` itself** (it references its own macros in comments)
3. Flag any macro that is defined but never referenced in section files
4. Check: are there model-specific macros for a model that's no longer discussed?
5. Check: do `make_figures.py` or other scripts use any "dead" macros? (if so, they're not dead)
6. Action: remove dead macros, or add a comment explaining why they're retained

```bash
python3 -c "
import re, os
defined = set(re.findall(r'\\\\newcommand\{\\\\(\w+)\}', open('paper/assets/macros.tex').read()))
used = set()
for root, dirs, files in os.walk('paper/v4'):
    for f in files:
        if f.endswith('.tex'):
            used.update(re.findall(r'\\\\(\w+)', open(os.path.join(root, f)).read()))
# Also check figures/scripts
for script in ['paper/figures/make_figures.py', 'paper/verify_numbers.py']:
    if os.path.exists(script):
        used.update(re.findall(r'\\\\(\w+)', open(script).read()))
dead = defined - used
print(f'Defined: {len(defined)}, Used: {len(defined & used)}, Dead: {len(dead)}')
for m in sorted(dead):
    print(f'  DEAD: \\\\{m}')
"
```

---

## Output Format

For each check, report:

```
Q[N]: [Question text]
ANSWER: [What was found]
STATUS: [CRITICAL / IMPORTANT / OK]
FIX: [What was done, or "no fix needed"]
```

After all 24 checks, summarize:
- Total issues found (by group)
- Critical issues fixed
- Remaining known limitations (acknowledged in paper)
- Compilation status (errors / overfull / undefined)
- verify_numbers.py status
- Dead macros count
- Derived calculation mismatches count
- Storytelling: setup-payoff closure gaps
- Readability: sentences >40 words, Flesch score range
- Definitions: undefined variables, undefined metrics, missing methodology context
