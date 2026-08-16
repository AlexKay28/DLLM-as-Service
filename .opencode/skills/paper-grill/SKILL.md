---
name: paper-grill
description: Use when reviewing a research paper for validity, consistency, and clarity. Runs a systematic 37-check audit across 7 dimensions — data integrity, narrative quality, mathematical correctness, presentation, completeness, and maintenance hygiene. Includes the arXiv storytelling-rigor trade-off: intuition-first methods, question-framed experiments, self-contained captions, real-world context, and automated readability/density evaluation. Triggers on phrases like "grill the paper", "review the paper", "check for contradictions", "find issues", or "audit paper".
---

# Paper Grill Skill

A systematic audit strategy for research papers that finds contradictions, unsupported claims, broken numbers, narrative inconsistencies, and readability barriers before a reviewer does. Run every check. For each, ask the question, search the paper, flag issues, and fix critical ones before moving on.

## What We Check vs What We Don't

**We check:**
- Numbers: macro values vs source JSONs, derived calculations, table cells
- Claims: measured vs projected, model identity, data attribution
- Narrative: story arc, setup-payoff closure, framing consistency
- Readability: sentence length, jargon density, acronym expansion, definitions
- Math: equation validity, crossover semantics, variable definitions
- Hygiene: dead macros, stale references, compilation, verify scripts
- Structural integrity: duplicate content, self-contradiction, arm mismatch, equation disclosure, citation accuracy, register, table cells, model name precision

**We don't check:**
- Whether the experimental design is sound (that's peer review, not grilling)
- Whether the cited related work is comprehensive
- Whether the chosen baselines are the right ones (we check if they're *disclosed*, not if they're *correct*)
- LaTeX formatting style (font choices, spacing, color schemes)
- Whether the figures look good (we check if captions match text, not aesthetics)

## Workflow

1. **Read** `macros.tex`, all section files, and the `results/*.json` artifacts in parallel
2. **Run Group A** (data integrity) first — fix any CRITICAL number mismatches before continuing
3. **Run Group B** (narrative) — assess storytelling, readability, definitions
4. **Run Group C** (math) — verify equations and derived calculations
5. **Run Groups D–F** (presentation, completeness, hygiene) — can run in parallel
6. **Run Group G** (structural integrity) — duplicate content, self-contradiction, arm mismatch, equation disclosure, citations, register, table cells, model names
7. **Compile** after all fixes
8. **Write fix-log** — record every change in `paper/v4/grill-fix-log.md` (see Fix-Log Format below)
9. **Run self-audit** — verify every fix was applied correctly (see Self-Audit Checklist below)
10. **Report** using the output format at the end

## Audit Strategy: 37 Checks in 7 Groups

---

### Group A: Data Integrity

#### Check 1: Number Consistency — Macros vs Calculated vs Text

**Question:** Does every number in the paper match its source data?

**Pass threshold:** 0 mismatches. Even a 0.1 difference between a macro and its JSON source is a fail.

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

**Pass thresholds:**
- Median sentence length: 15–25 words
- Sentences >40 words: ≤2 per section (equations + where-blocks are exempt — inspect manually)
- Sentences >60 words: 0 (after excluding equation blocks)
- Flesch Reading Ease: 30–50 per section (main text); appendices may be denser (20+)
- Bare acronyms before expansion: 0
- Consecutive passive-voice sentences: ≤3
- Hedging words per sentence: ≤1

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

**Pass thresholds:**
- Equations without where-blocks: 0 (within 1000-char window; align environments whose variables are defined in a preceding equation are OK)
- Undefined metrics: 0 (F1, CV, max/median, p50, p99 must be defined or explained before first numeric use)
- Results cited without model identity: 0 (every cited measurement must name the model)
- Results cited without n: 0 (every cited measurement must state n or reference the default \nPerCell)
- Ambiguous antecedents ("this approach", "that model"): 0

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

#### Check 25: Forward-Use of Abbreviations and Paper-Specific Terms Before Definition

**Question:** Are any abbreviations or paper-specific terms used before they are defined?

A reader hitting the introduction should not encounter `L*`, `batch-1 regime`, `DG`, `AR`, `canvas`, `K`, `λ*`, `p99`, `CV`, `MoE`, or any other specialized token without first seeing its straight definition in plain language. The introduction sets up the paper's vocabulary; if it borrows from later sections without defining, the reader is lost before the story begins.

**Pass threshold:** 0 forward uses. Every abbreviation and paper-specific term must be defined in plain English **at or before** its first occurrence in every section. A definition that lives in §3 does not cover a use in §1.

1. **Abbreviation-first scan:** For every abbreviation found in the paper (two-or-more capital letters, or mixed-case shorthand like `L*`), locate its first occurrence in the full document. Verify that the expansion appears in the same sentence or the sentence immediately preceding. Flag any abbreviation whose first use is bare.
2. **Paper-specific term scan:** Build a glossary of terms this paper invents or repurposes: `L*`, `λ*`, `batch-1 regime`, `canvas`, `denoising step`, `K` (as a step count), `serial mode`, `packed mode`, `first-object-wins`, `natural output length`, `ramble rate`, `bounded deadline`, `cost shape`, `staircase`, `floor`. For each, find its first occurrence in each section and verify a plain-English definition precedes or accompanies it. Flag forward uses.
3. **Cross-section independence:** A term defined in §2 and used in §1 is a forward use — the definition must appear in §1 (or the abstract) too, or the use must be removed. The abstract and introduction are read first and must be self-contained.
4. **Math notation before definition:** Symbols like `L*`, `K=8`, `B` (blocks), `n_c` (canvas count) must not appear in prose before the equation or sentence that defines them. Flag any symbol that appears in the introduction or abstract without a preceding definition.
5. **Abstract-specific rule:** The abstract is the widest-read part of the paper. It must contain **zero** undefined abbreviations and **zero** undefined paper-specific terms. Model names (e.g., DiffusionGemma-26B-A4B) are permitted; shorthands (AR, DG, MoE) are not, unless spelled out in the same sentence.

```bash
# Build the glossary and scan for first-use-before-definition
python3 -c "
import re, os

# Terms to check (add more as the paper evolves)
TERMS = {
    'abbreviations': ['AR', 'DG', 'MoE', 'KV', 'F1', 'CV', 'p50', 'p99', 'p99.9', 'SLA', 'IQR', 'BF16'],
    'paper_specific': ['L*', 'lambda*', 'batch-1', 'batch-1 regime', 'canvas', 'K=8', 'K steps',
                       'denoising step', 'serial mode', 'packed mode', 'first-object-wins',
                       'natural output length', 'ramble rate', 'bounded deadline',
                       'cost shape', 'staircase', 'floor', 'n_c'],
}

# Read sections in document order
section_order = sorted(os.listdir('paper/v4/sections'))
all_text = {}
for f in section_order:
    if f.endswith('.tex'):
        all_text[f] = open(f'paper/v4/sections/{f}').read()

# Check abstract separately
abstract = open('paper/v4/main.tex').read()
abs_match = re.search(r'\\\\begin\{abstract\}(.*?)\\\\end\{abstract\}', abstract, re.DOTALL)
abs_text = abs_match.group(1) if abs_match else ''

print('=== ABSTRACT ===')
for category, terms in TERMS.items():
    for term in terms:
        if term in abs_text:
            # Check if defined in the same sentence or the one before
            sentences = re.split(r'(?<=[.!?])\s+', abs_text)
            for i, s in enumerate(sentences):
                if term in s:
                    # Check if the term is expanded/defined in this or preceding sentence
                    context = (sentences[i-1] if i > 0 else '') + ' ' + s
                    expanded = any(w in context.lower() for w in
                        ['autoregressive', 'block diffusion', 'block-diffusion',
                         'mixture of experts', 'key-value', 'field-f1', 'coefficient',
                         'percentile', 'service level', 'interquartile', 'bfloat16'])
                    if not expanded and term in ('AR', 'DG', 'MoE', 'KV', 'BF16'):
                        print(f'  BARE ABBR in abstract: {term!r} — define or spell out')
                    break

print()
print('=== SECTION SCAN (first occurrence per section) ===')
for fname, text in all_text.items():
    issues = []
    for term in TERMS['paper_specific'] + TERMS['abbreviations']:
        if term in text:
            # Find the first occurrence
            idx = text.find(term)
            # Get the 200-char window before it
            before = text[max(0, idx-200):idx]
            # Check if a definition-like phrase appears before
            defined = any(d in before.lower() for d in [
                'is defined as', 'we define', 'denotes', 'where ', 'let ', 'i.e.,',
                'that is,', 'we call', 'we term', 'refers to',
                'autoregressive', 'block diffusion', 'block-diffusion',
                'mixture of experts', 'key-value', 'percentile',
            ])
            if not defined:
                issues.append(term)
    if issues:
        print(f'  {fname}: forward-use before definition: {issues}')
"
```

**Common fixes:**
- Spell out the abbreviation at first use in each section where it appears: "autoregressive (AR)" → then use AR freely in that section.
- For paper-specific terms (L*, batch-1 regime), add a one-clause definition at first use: "the crossover length $L^*$ (where the two cost curves meet)".
- In the abstract, prefer full terms over abbreviations entirely.

#### Check 26: Intuition Before Formalism — The arXiv Dual-Layer Test

**Question:** Does every methods subsection open with plain-English intuition before equations arrive?

The ideal arXiv trade-off is **dual-layer**: high-level conceptual intuition *before* formal definitions. A reader who doesn't parse equations should still understand the mechanism from the first 1–2 sentences. If the first thing a reader sees is `T = a + b \cdot f(L)` without a plain-English preamble, the section fails this check.

**Pass threshold:** Every subsection containing an equation must have ≥1 sentence of intuition before the first equation environment.

1. **Intuition-first check:** For every `\subsection` or `\paragraph` that contains an `\begin{equation}` or `\begin{align}`, read the text *before* the equation. Is there a plain-English sentence explaining what the equation says in words? Flag equations that appear with no conceptual preamble.
2. **Story → rigor → detail structure:** Check that the flow is: (1) intuition (what does this do?), (2) formal definition (equation + where-block), (3) implementation detail (what values were measured). Flag sections that skip step 1 or 3.
3. **Equation purpose check:** After every equation, is there a sentence explaining *what this equation tells us* (not just *what the variables are*)? The where-block defines variables; the sentence after it should state the finding. E.g., "This means AR's cost grows with every token, while DG's is bounded by canvas size."
4. **Analogy presence:** Does the paper use at least one real-world analogy for its core mechanism? (e.g., "AR is a metered taxi; DG is a subway.") Flag papers that are 100% formal with zero analogies — they fail the arXiv trade-off.

```bash
# Find equations and check what precedes them
python3 -c "
import re, os
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    for m in re.finditer(r'\\\\begin\{(equation|align)\*?\}', text):
        # Get 300 chars before the equation
        before = text[max(0, m.start()-300):m.start()]
        # Strip LaTeX commands for readability check
        clean = re.sub(r'\\\\\w+(\[[^\]]*\])?(\{[^}]*\})?', '', before)
        clean = re.sub(r'%.*', '', clean).strip()
        sentences = [s.strip() for s in clean.split('.') if len(s.strip()) > 15]
        has_intuition = len(sentences) > 0 and not sentences[-1].startswith('where')
        if not has_intuition:
            print(f'{f}: equation at char {m.start()} — no intuition sentence before it')
"
# Check for analogies
grep -n 'like a\|think of\|analogy\|equivalent of\|to put this\|in human terms\|in cost terms\|to make this concrete' paper/v4/sections/*.tex
```

#### Check 27: Question-Framed Experiments — "Why" Not "What"

**Question:** Are tables and experiments framed by the question they answer, and does the text after each table explain *why* the pattern exists?

A table presents numbers; the text must explain the mechanism. If the text after a table merely restates the numbers ("Table 1 shows AR is 15.6× slower"), it fails. The text must answer *why*: "AR latency is flat from 128 tokens because EOS stops generation; DG latency steps by ~120ms per canvas because each canvas is a fresh denoising pass."

**Pass threshold:** Every table must be (a) preceded by a question or setup, and (b) followed by a "why" explanation that names the mechanism, not just the numbers.

1. **Pre-table question:** Before every `\begin{table}`, is there a sentence stating what question the table answers? (e.g., "How does serial latency compare at the MoE-matched pair?") Flag tables that appear with no framing.
2. **Post-table mechanism:** After every `\end{table}`, is there a sentence explaining *why* the pattern exists? Flag tables followed by restated numbers without causal explanation.
3. **Counterintuitive result explanation:** If a table contains a counterintuitive result (e.g., a slower model that's cheaper per event), is the mechanism explained? Flag unexplained surprises.
4. **Real-world context:** Are key numeric results translated into real-world context (cost in dollars, time in human-perceptible units, energy in kWh)? Flag papers that present only abstract numbers (ms, J, F1) without any translation. At minimum: the abstract and conclusion should each contain at least one cost or time analogy.
5. **Currency translation:** For cost/performance claims, is there at least one translation to dollars? (e.g., "$0.0005 per event" or "$26,000/year gap"). Flag papers with cost tables but no dollar-context in the text.

```bash
# Find tables and check what follows them
python3 -c "
import re, os
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    for m in re.finditer(r'\\\\end\{table\}', text):
        after = text[m.end():m.end()+500]
        # Strip LaTeX for readability
        clean = re.sub(r'\\\\\w+(\[[^\]]*\])?(\{[^}]*\})?', '', after)
        clean = re.sub(r'%.*', '', clean).strip()
        # Check for 'because', 'due to', 'since', 'the reason', 'this is because'
        has_mechanism = any(w in clean.lower() for w in ['because', 'due to', 'since', 'the reason', 'this is', 'the pattern'])
        if not has_mechanism:
            print(f'{f}: table ends at char {m.end()} — no mechanism explanation within 500 chars after')
"
# Check for real-world context
grep -n 'cost\|dollar\|\\$\|kWh\|human\|blink\|second\|minute\|hour\|day\|year\|per million\|per event' paper/v4/sections/*.tex | grep -v 'caption\|label\|begin\|end'
# Check for question-framing before tables
grep -B2 'begin{table}' paper/v4/sections/*.tex
```

#### Check 28: Self-Contained Figure Captions

**Question:** Can a reviewer understand the finding from the caption alone, without reading the body text?

The ideal arXiv figure caption tells a micro-story: what is shown, what the pattern is, and why it matters. A caption that says "Latency vs output length" fails; a caption that says "AR latency grows linearly while DG is a staircase — the gap widens because AR has no floor and DG is bounded" passes.

**Pass threshold:** Every figure caption must contain (a) what is plotted, (b) the finding/pattern, and (c) the mechanism or "why".

1. **Three-part caption test:** For every `\caption{}`, check that it contains: (a) what the axes/panels show, (b) what the reader should notice (the pattern), and (c) why the pattern exists. Flag captions that only describe (a) without (b) and (c).
2. **Standalone readability:** Read each caption in isolation. Does it make sense without the surrounding text? Flag captions that reference undefined context ("as described above", "see Section X") without restating the key finding.
3. **Panel labeling:** For multi-panel figures, does each panel (A, B, ...) have its own micro-finding in the caption? Flag panel descriptions that only state what's plotted, not what it shows.
4. **Data source disclosure:** Does the caption state which model/arm the data comes from? (e.g., "AR arm is dense AR-3B baseline"). Flag captions that don't disclose the model when the paper uses multiple models.
5. **Numbers in captions:** Does the caption cite at least one key number? (e.g., "15.6× at L=256"). Flag captions with zero numbers — they're too vague.

```bash
# Extract and evaluate all captions
python3 -c "
import re, os
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    for m in re.finditer(r'\\\\caption\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', text):
        cap = m.group(1)
        # Strip nested LaTeX
        clean = re.sub(r'\\\\\w+(\[[^\]]*\])?(\{[^}]*\})?', '', cap).strip()
        words = clean.split()
        has_number = any(re.search(r'\d', w) for w in words)
        has_pattern = any(w in clean.lower() for w in ['grow', 'flat', 'jump', 'widens', 'narrows', 'saturates', 'collapses', 'staircase', 'linear', 'plateau', 'ratio'])
        has_mechanism = any(w in clean.lower() for w in ['because', 'since', 'due to', 'while', 'whereas', 'unlike', 'contrast'])
        has_model = any(w in cap for w in ['AR-3B', 'AR-30B', 'AR-1.5B', 'DG', 'dense', 'baseline'])
        status = 'OK' if (has_number and has_pattern) else 'CHECK'
        print(f'{f}: [{status}] {len(words)}w  num={has_number} pattern={has_pattern} mechanism={has_mechanism} model={has_model}')
        print(f'  {clean[:150]}')
        print()
"
```

---

### Group B-2: Automated Readability & Density Evaluation

#### Check 29: Readability and Expression Density (Automated)

**Question:** Does the paper pass automated readability and expression-density thresholds?

This check runs `paper/eval_readability.py`, which computes Flesch Reading Ease, sentence length distribution, passive voice clusters, hedging, number density, bracket density, and words-per-number for every section. It strips LaTeX commands, equation/table environments, and footnotes before analysis, so only prose is measured.

**Pass thresholds (all enforced by the script):**
- Flesch Reading Ease: 18–55 for main text, 20+ for appendices
- Median sentence length: 10–28 words
- Sentences >40 words: ≤2 per section (excl. equation/table envs)
- Sentences >60 words: 0 per section
- Passive voice clusters: ≤3 consecutive
- Hedging ratio: ≤0.5 per sentence
- Number density: ≤4.0/100w main text, ≤5.0/100w any section
- Bracket density: ≤1.5/100w main text (excludes citation brackets)
- Words per number: ≥25 in main text
- Abstract inline numbers: ≤3

```bash
# Run the full evaluation (exit 0 = pass, exit 1 = fail)
python3 paper/eval_readability.py

# Readability only
python3 paper/eval_readability.py --read

# Density only
python3 paper/eval_readability.py --density

# Machine-readable JSON output
python3 paper/eval_readability.py --json
```

The script reports per-section metrics with PASS/FAIL status and the applicable threshold. If any check fails, the script exits with code 1. The grill agent should fix the failing section and re-run until all checks pass.

**Common fixes for failures:**
- Flesch too low (dense): shorten sentences, use simpler words, split long clauses
- Number density too high: move numbers to tables/equations, use words ("fifteen times" not "15.6×")
- Bracket density too high: expand acronyms inline, move context to footnotes/captions
- Sentences >60 words: split at semicolons or em-dashes
- Abstract numbers >3: replace inline ratios with word-form ("fifteenfold")

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

#### Check 12: Cost-Per-Event Table Consistency

**Question:** Are all cost calculations correct and complete?

**Pass threshold:** Every events/hr and $/event value must recompute to within 1% of the stated value.

1. Verify: events/hr = 3600000 / p50_ms (not 3600 / p50_seconds — unit trap)
2. Verify: $/event = hourly_rate / events_hr
3. Check: are ALL models in the table (AR-30B, AR-1.5B, DG)?
4. Check: is the cost ratio R_cost calculated from the right pair? (if the equation says 0.0009/0.0005, verify those are the right rows)
5. Verify: does the co-residency argument use the right model (smallest AR)?
6. Check: are duty cycle calculations correct (25% = events_hr × 0.25)?
7. Check: if the table says "AR-30B serial" but the events/hr implies a different p50, flag it

```bash
python3 -c "
gpu_rate = 4.62
# AR-30B serial: p50 ~5900ms
ar30_events_hr = 3600000 / 5900
ar30_cost = gpu_rate / ar30_events_hr
print(f'AR-30B: {ar30_events_hr:.0f} ev/hr, \${ar30_cost:.4f}/ev')
# AR-1.5B: p50 ~665.7ms
ar1b_events_hr = 3600000 / 665.7
ar1b_cost = gpu_rate / ar1b_events_hr
print(f'AR-1.5B: {ar1b_events_hr:.0f} ev/hr, \${ar1b_cost:.4f}/ev')
# DG: p50 ~379ms
dg_events_hr = 3600000 / 379
dg_cost = gpu_rate / dg_events_hr
print(f'DG: {dg_events_hr:.0f} ev/hr, \${dg_cost:.4f}/ev')
"
```

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

**Pass threshold:** 0 dead macros (macros used only in `macros.tex` comments, or only in figure scripts, count as used).

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

### Group G: Structural Integrity and Reproducibility

#### Check 30: Duplicate Content Across Sections

**Question:** Is any block of text (paragraph, list, table) duplicated — or near-duplicated — across sections?

Duplicate content signals a structural problem: either a section was copy-pasted during restructuring and never cleaned up, or two sections serve the same purpose. Both confuse the reader about which version is authoritative.

**Pass threshold:** 0 duplicated paragraphs (≥3 consecutive sentences matching another section) outside deliberate cross-references.

1. **Fuzzy paragraph matching:** Extract every paragraph (text block separated by blank lines) from each section. Normalize (strip LaTeX commands, lowercase, collapse whitespace). Compare every pair of paragraphs across different sections. Flag pairs with >60% token overlap.
2. **Duplicate lists:** The same enumerated/itemized list appearing in two sections is a strong signal of duplication. Compare list item sets across sections.
3. **Section role overlap:** If two sections have the same title or the same opening sentence, one is likely redundant.
4. **Cross-reference vs duplicate:** A deliberate cross-reference ("as stated in §X") is fine; a full re-statement of the same content is a duplicate.

```bash
python3 -c "
import re, os
from difflib import SequenceMatcher

sections = {}
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    # Split into paragraphs
    paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    # Normalize for comparison
    normalized = []
    for p in paras:
        clean = re.sub(r'\\\\\w+(\[[^\]]*\])?(\{[^}]*\})?', '', p)
        clean = re.sub(r'%.*', '', clean).lower().strip()
        clean = re.sub(r'\s+', ' ', clean)
        normalized.append(clean)
    sections[f] = list(zip(paras, normalized))

# Compare every pair of sections
files = list(sections.keys())
for i in range(len(files)):
    for j in range(i+1, len(files)):
        for p1, n1 in sections[files[i]]:
            for p2, n2 in sections[files[j]]:
                if len(n1) < 80 or len(n2) < 80: continue
                ratio = SequenceMatcher(None, n1, n2).ratio()
                if ratio > 0.6:
                    print(f'DUPLICATE ({ratio:.0%} match):')
                    print(f'  {files[i]}: {p1[:100]}...')
                    print(f'  {files[j]}: {p2[:100]}...')
                    print()
"
```

#### Check 31: Self-Contradiction Between Threats and Headline Claims

**Question:** Does the paper's own threats section contradict a claim stated as fact elsewhere?

The most dangerous inconsistency is one the paper itself acknowledges but doesn't propagate. If the threats section says "these numbers are unverified" but the abstract/intro/conclusion state them as fact, the paper contradicts itself.

**Pass threshold:** 0 contradictions. Every claim flagged as uncertain in the threats section must be hedged wherever it appears.

1. **Threats claim extraction:** Read the threats section. For every sentence that qualifies, limits, or retracts a claim (keywords: "not yet", "unverified", "requires re-measurement", "indicative", "not controlled", "not tested", "not measured", "upper bound"), record the claim and the qualification.
2. **Headline scan:** For each qualified claim, search the abstract, intro contributions, and conclusion for the same claim. Is it stated with the same qualification, or as flat fact? Flag any unqualified restatement.
3. **Cross-section propagation:** If the threats section says "X is unverified," every use of X in the body must carry a hedge (e.g., "unverified", "projected", "pending re-measurement").
4. **Direction consistency:** If the threats section says a number "plausibly" inflates the diffusion advantage, the body text must not present that number as a lower bound.

```bash
# Extract threat qualifications and search for unqualified restatements
python3 -c "
import re, os

threats = open('paper/v4/sections/07_threats.tex').read()
# Find qualifying phrases
qualifiers = re.findall(r'(not yet|unverified|requires? re-measurement|indicative|not controlled|not tested|not measured|upper bound|inherited from|have not been)', threats, re.IGNORECASE)
print(f'Threats section has {len(qualifiers)} qualifying phrases')

# Extract claims near each qualifier
sentences = re.split(r'(?<=[.!?])\s+', re.sub(r'\\\\\w+(\{[^}]*\})?', '', threats))
flagged = [s.strip() for s in sentences if any(q in s.lower() for q in ['not yet', 'unverified', 'requires re-measurement', 'indicative', 'not controlled', 'not tested', 'not measured', 'upper bound', 'inherited from', 'have not been'])]
print('Qualified claims in threats:')
for s in flagged:
    print(f'  - {s[:150]}')
print()

# Check abstract, intro, conclusion for numbers cited as fact
for f in ['paper/v4/main.tex', 'paper/v4/sections/01_introduction.tex', 'paper/v4/sections/08_conclusion.tex']:
    text = open(f).read()
    # Look for headline numbers without hedge words
    for pattern in [r'15\.6', r'8\.0\\\\times', r'eightfold', r'fifteenfold', r'77\\\\times', r'seventy']:
        matches = re.findall(f'.{{0,80}}{pattern}.{{0,80}}', text)
        for m in matches:
            has_hedge = any(h in m.lower() for h in ['unverified', 'projected', 'pending', 'upper bound', 'not yet'])
            if not has_hedge:
                print(f'  UNQUALIFIED in {f}: ...{m.strip()[:100]}...')
"
```

#### Check 32: Quality-vs-Latency Arm Mismatch

**Question:** Are quality (F1) and latency (ms) numbers for the same comparison drawn from the same model?

A common pattern: the paper claims "the two models are quality-matched" using F1 from model A, then presents latency from model B as if the comparison were one-model-vs-one-model. This is invisible unless you trace which model produced each metric.

**Pass threshold:** 0 mismatches. Every table or claim that pairs quality and latency for a single comparison must use the same model for both axes.

1. **Arm tracing:** For every comparison table or claim, identify which model produced the quality metric and which produced the latency metric. If they differ, flag it.
2. **Matched-pair verification:** If the paper says "matched pair" or "fair comparison," verify that both quality and latency come from the same pair of models.
3. **Inline disclosure:** If arms must differ (e.g., quality was only measured on a smaller model), is this disclosed at the point of the claim, not only in the threats section?

```bash
python3 -c "
import re, os

# Search for quality claims and latency claims, trace the model
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    # Find paragraphs mentioning both F1/quality and latency/ms together
    paras = text.split('\n\n')
    for p in paras:
        has_quality = bool(re.search(r'F1|field.f1|quality|equivalen', p, re.IGNORECASE))
        has_latency = bool(re.search(r'ms|latency|p50|p99|faster|slower', p, re.IGNORECASE))
        if has_quality and has_latency:
            models_mentioned = re.findall(r'AR-3B|AR-30B|AR-1.5B|Qwen2.5-3B|Qwen3-30B|dense', p)
            if len(set(models_mentioned)) > 1:
                clean = re.sub(r'\\\\\w+(\{[^}]*\})?', '', p)[:200]
                print(f'{f}: MIXED ARMS in quality+latency paragraph:')
                print(f'  Models: {set(models_mentioned)}')
                print(f'  Text: {clean.strip()}...')
                print()
"
```

#### Check 33: Equation Simplification Disclosure

**Question:** Does every equation disclose known simplifications at the point of use?

If the paper knows a cost law depends on two variables but writes it as a one-variable equation, the simplification must be disclosed at the equation, not deferred to a later paragraph or a footnote in an appendix.

**Pass threshold:** 0 undisclosed simplifications. Every equation that omits a known variable must state the omission and its scope.

1. **Known-variable audit:** For each equation, check whether the paper elsewhere acknowledges variables that the equation doesn't include. If the text says "cost depends on X and Y" but the equation only has X, flag it.
2. **Scope disclosure:** The equation must state its validity scope (e.g., "valid at the reference prompt length of 1077 tokens; see Appendix X for the two-variable form").
3. **Direction of simplification:** Does the simplification favor one arm? If the omitted variable makes one architecture look better, the disclosure must say so.

```bash
python3 -c "
import re, os

# Find all equations and check for two-variable/two-variable mentions nearby
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    # Find equation environments
    for m in re.finditer(r'\\\\begin\{(equation|align)\*?\}(.*?)\\\\end\{\1\*?\}', text, re.DOTALL):
        eq = m.group(2)
        # Check if the equation has only one variable (L_out) but paper mentions L_in
        has_l_out = 'L_{\\\\mathrm{out}}' in eq or 'L_{out}' in eq or 'L_out' in eq
        has_l_in = 'L_{\\\\mathrm{in}}' in eq or 'L_{in}' in eq or 'L_in' in eq
        if has_l_out and not has_l_in:
            # Check if input length is mentioned elsewhere in the paper
            full_text = open(f'paper/v4/sections/{f}').read()
            input_mentions = re.findall(r'input length|input slope|prompt length|prefill slope|two.variable|L_\\\\mathrm\{in\}', full_text, re.IGNORECASE)
            if input_mentions:
                print(f'{f}: equation has L_out only, but section mentions input length {len(input_mentions)} times')
                print(f'  Equation: {eq.strip()[:150]}')
                print(f'  → Disclose simplification at the equation')
                print()
"
```

#### Check 34: Stale or Wrong Citations

**Question:** Does every citation match the model, method, or tool it's attached to?

Citations drift during model switches. A paper that switches from Qwen2.5 to Qwen3 may forget to update the `\cite{}`, leaving a Qwen2.5 reference on a Qwen3 model name.

**Pass threshold:** 0 mismatches. Every `\cite{}` must reference the correct version of the model/method/tool it's attached to.

1. **Citation-model pairing:** For every model name in the text, check that the adjacent `\cite{}` references the correct model family and version.
2. **Version drift:** If the text says "Qwen3-30B-A3B" but the citation key is `qwen25`, that's a mismatch. Build a map of model-name → expected-citation-key and verify.
3. **Tool version drift:** If the text says "vLLM 0.25.1" but cites `vllm2023` (the 2023 paper, not the 0.25.1 release), verify the citation refers to the right thing.

```bash
python3 -c "
import re, os

# Map of model names to expected citation fragments
MODEL_CITATIONS = {
    'Qwen3': 'qwen3',
    'Qwen2.5': 'qwen25',
    'DiffusionGemma': 'diffusiongemma',
    'LLaDA': 'llada',
    'Drain': 'drain',
}

for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    # Find model name + adjacent citation
    for model, expected_key in MODEL_CITATIONS.items():
        for m in re.finditer(rf'{model}\S*~?\\\\cite\{{([^}}]+)\}}', text):
            cite_key = m.group(1)
            if expected_key not in cite_key:
                print(f'{f}: MODEL-CITATION MISMATCH')
                print(f'  Model: {model}, citation: \\\\cite{{{cite_key}}}')
                print(f'  Expected key containing: {expected_key}')
                print()
        # Also check the reverse: citation without the right model
        for m in re.finditer(rf'\\\\cite\{{{expected_key}\}}}.*?{model}', text[:m.start()+200] if m else ''):
            pass  # forward reference is fine
"
```

#### Check 35: Register Consistency and Proofreading

**Question:** Is the writing register (formal/informal) consistent across sections, and are there stray punctuation errors?

A paper that uses contractions in the conclusion but not in the introduction has a register shift. Stray commas, missing commas, and doubled spaces are proofreading errors that erode credibility.

**Pass threshold:** 0 register shifts (contractions in one section but not others); 0 stray punctuation.

1. **Contraction scan:** Find all contractions ("it's", "can't", "doesn't", "that's", "isn't", "won't"). If they appear in some sections but not others, that's a register shift. Either use contractions throughout or not at all.
2. **Stray punctuation:** Find " ," (space before comma), " ." (space before period), ",," (doubled comma), "  " (doubled space between words, not after periods).
3. **Em-dash consistency:** If the paper uses "---" (em-dash) in some places and "--" (en-dash) in others for the same purpose, flag it.
4. **Missing commas after introductory clauses:** Sentences starting with "At eight hops," or "In serial mode," need the comma; flag its absence.

```bash
python3 -c "
import re, os

contractions_by_section = {}
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    clean = re.sub(r'\\\\\w+(\[[^\]]*\])?(\{[^}]*\})?', '', text)
    clean = re.sub(r'%.*', '', clean)
    
    contractions = re.findall(r\"\\b(it's|can't|doesn't|that's|isn't|won't|don't|aren't|wasn't|weren't|hasn't|haven't|wouldn't|couldn't|shouldn't|didn't)\", clean, re.IGNORECASE)
    if contractions:
        contractions_by_section[f] = contractions
        print(f'{f}: {len(contractions)} contractions: {contractions[:5]}')

if len(set(contractions_by_section.keys())) > 0 and len(set(contractions_by_section.keys())) < len([f for f in os.listdir('paper/v4/sections') if f.endswith('.tex')]):
    print()
    print('REGISTER SHIFT: contractions in some sections but not others')
    print(f'  Sections with contractions: {list(contractions_by_section.keys())}')

print()
print('=== Stray punctuation ===')
for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    issues = []
    if re.search(r' ,', text): issues.append('space-before-comma')
    if re.search(r' \.', text) and not re.search(r'e\.g\.', text): issues.append('space-before-period')
    if re.search(r',,', text): issues.append('doubled-comma')
    if re.search(r'  [a-z]', text): issues.append('doubled-space')
    if issues:
        print(f'  {f}: {issues}')
"
```

#### Check 36: Degenerate Table Cells

**Question:** Are there empty, placeholder, or degenerate cells in any table?

A table cell that contains only a comma, an empty string, or "N/A" without explanation is a rendering artifact or a missing value. The reader shouldn't have to guess.

**Pass threshold:** 0 degenerate cells. Every cell must contain a value, a macro, or an explicit "—" (em-dash) with a caption note explaining what it means.

1. **Empty cell scan:** Parse every `\begin{tabular}` environment. For each row, split by `&`. Flag cells that are empty, contain only whitespace, or contain only a comma/punctuation.
2. **Placeholder detection:** Flag cells containing "TODO", "TBD", "XXX", "?", or a bare comma.
3. **Asymmetry check:** If a row has N columns but a cell produces fewer than expected values (missing `&`), flag it.

```bash
python3 -c "
import re, os

for f in sorted(os.listdir('paper/v4/sections')):
    if not f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{f}').read()
    for m in re.finditer(r'\\\\begin\{tabular[x]?\}\{[^}]*\}(.*?)\\\\end\{tabular[x]?\}', text, re.DOTALL):
        table = m.group(1)
        rows = [r.strip() for r in table.split('\\\\\\\\') if r.strip()]
        for i, row in enumerate(rows):
            cells = [c.strip() for c in row.split('&')]
            for j, cell in enumerate(cells):
                clean = re.sub(r'\\\\\w+(\{[^}]*\})?', '', cell).strip()
                if not clean or clean in (',', '.', '-', '?'):
                    print(f'{f}: row {i}, col {j}: DEGENERATE CELL: {cell!r}')
                if clean in ('TODO', 'TBD', 'XXX', 'N/A'):
                    print(f'{f}: row {i}, col {j}: PLACEHOLDER: {cell!r}')
"
```

#### Check 37: Model Name Precision and Reproducibility

**Question:** Do the model names in the paper match the HuggingFace IDs exactly enough to reproduce the results?

A model called "DiffusionGemma-26B-A4B" in the paper might be `google/diffusiongemma-26b-a4b-it` on HuggingFace — the missing `-it` (instruction-tuned) suffix changes the model. The paper must use names precise enough for a reader to load the exact checkpoint.

**Pass threshold:** 0 ambiguous model names. Every model name in the text must either (a) match a HuggingFace ID exactly, or (b) be accompanied by the full HF ID in parentheses or a footnote.

1. **Name-to-ID mapping:** Build a map of every model name used in the paper to its HuggingFace ID (from `results/*.json` provenance blocks or `models/baselines/*.py` defaults). Verify the paper's name is unambiguous.
2. **Suffix check:** Common suffixes that change behavior: `-it` (instruction-tuned), `-Instruct`, `-chat`, `-base`. If the paper drops a suffix, flag it.
3. **Version check:** If the paper says "Qwen2.5-3B" but the provenance says "Qwen/Qwen2.5-3B-Instruct", flag the missing `-Instruct`.
4. **Case sensitivity:** Model IDs on HuggingFace are case-sensitive. Flag case mismatches between the paper and the HF ID.

```bash
python3 -c "
import json, os, re

# Build model name -> HF ID map from provenance blocks
hf_map = {}
for root, dirs, files in os.walk('results'):
    for f in files:
        if f.endswith('.json') and 'provenance' not in f:
            try:
                d = json.load(open(os.path.join(root, f)))
                prov = d.get('provenance', {})
                if isinstance(prov, dict) and prov.get('model_name'):
                    hf_map[prov['model_name']] = f
            except: pass

# Check paper text for model names
for sec_f in sorted(os.listdir('paper/v4/sections')):
    if not sec_f.endswith('.tex'): continue
    text = open(f'paper/v4/sections/{sec_f}').read()
    for hf_id, artifact in hf_map.items():
        # Extract the short name from the HF ID
        short_name = hf_id.split('/')[-1]
        if short_name in text:
            # Check if the full HF ID appears nearby
            idx = text.find(short_name)
            context = text[max(0,idx-200):idx+len(short_name)+200]
            if hf_id not in context:
                print(f'{sec_f}: MODEL NAME IMPRECISE')
                print(f'  Paper uses: {short_name}')
                print(f'  HF ID: {hf_id}')
                print(f'  → Add full HF ID or verify suffix')
                print()
" 2>/dev/null || echo 'Run from repo root'
```

---

## False-Positive Catalog

These patterns look like issues but are NOT. Skip them to avoid wasting time:

| Pattern | Looks like | Why it's OK |
|---------|-----------|-------------|
| Equation + where-block parsed as one "sentence" | Readability check flags 90-word "sentence" | The where-block is a variable list, not prose. Exclude equation environments before sentence-length analysis. |
| `align` environment without where-block | Definitions check flags missing where-block | If the `align` follows an `equation` that has a where-block defining shared variables, the `align` inherits those definitions. Check the preceding 1000 chars. |
| Macro used only in `macros.tex` comments | Dead macro check flags it | Comments in `macros.tex` that reference macros are self-documentation, not usage. But the macro IS dead if no section file uses it. |
| Hardcoded numbers in section files | "No numeral in section file" rule violation | Derived ratios (15.6×), structural constants (12 fields, K=8), and table values may be hardcoded for readability. Only raw measurements must be macros. |
| Different F1 values for "same" config | Number mismatch | Same model + config measured in different runs (n=200 vs n=300) will differ by ~0.001–0.003. Flag only if the difference exceeds the stated SE. |
| Superseded values cited in discrepancy discussion | Stale reference | If the paper explicitly says "old value was X, new value is Y" in an appendix, the old value is intentional. Don't flag it. |
| Flesch Reading Ease >50 in appendices | Readability too informal | Appendices are denser by nature. Target 20+ for appendices, 30–50 for main text. |
| `arHdfs`/`arBgl` macros dead but values in text | Dead macro | Values may be hardcoded in the section file. The macro is dead; the value is live. Remove the macro, keep the text. |

---

## Fix-Log Format

After applying fixes, write `paper/v4/grill-fix-log.md` with this structure:

```markdown
# Grill Fix Log — [date]

## Pass [N]

| # | File | Change | Check | Severity |
|---|------|--------|-------|----------|
| 1 | paper/assets/macros.tex | `canvasOneTwentyEightBase` 385.5 → 368.5 | Check 1 | CRITICAL |
| 2 | sections/02_cost_laws.tex | K=32 F1 0.858 → 0.862 | Check 1 | CRITICAL |
| 3 | sections/15_appendix_distributions.tex | AR-30B → AR-3B (dense) | Check 4 | CRITICAL |

## Compilation: [errors/overfull/undefined counts]
## Dead macros removed: [count]
## Self-audit: [pass/fail per item]
```

The fix-log serves the same purpose as paper-deck's `generation-log.md`: every change is traceable to a check and a severity, so a future pass can verify the fix was applied and didn't introduce new issues.

---

## Self-Audit Checklist

After all fixes are applied and the PDF is recompiled, verify:

### Data integrity
- [ ] Every macro value matches its source JSON (re-run Check 1 verification)
- [ ] No hardcoded numbers in sections that should be macros (raw measurements only)
- [ ] All derived calculations recompute correctly (re-run Check 16 script)
- [ ] Dead macros: 0 (re-run Check 21 script)

### Narrative
- [ ] Abstract, intro, conclusion cite the same numbers (re-run Check 14)
- [ ] Crossover claims are integer-safe (re-run Check 18)
- [ ] No contradictory framing (re-read abstract + conclusion first/last sentences)
- [ ] No abbreviations or paper-specific terms used before their definition (re-run Check 25)
- [ ] Every equation has intuition before it (re-run Check 26)
- [ ] Every table has a "why" explanation after it (re-run Check 27)
- [ ] Every figure caption is self-contained (re-run Check 28)
- [ ] At least one real-world analogy in the paper (Check 26)
- [ ] At least one dollar-cost translation in abstract + conclusion (Check 27)
- [ ] `eval_readability.py` passes (exit 0)

### Compilation
- [ ] 0 errors, 0 overfull, 0 undefined (recompile from scratch)
- [ ] `verify_numbers.py` passes (run it)
- [ ] Fix-log written and complete

### Regression check
- [ ] No new dead macros introduced (removing a macro may orphan another)
- [ ] No new undefined references (editing sections may break \ref)
- [ ] No new stale numbers (fixing one number may reveal a dependent number that also needs updating)

### Structural integrity
- [ ] No duplicated paragraphs across sections (re-run Check 30)
- [ ] No self-contradiction between threats and headline claims (re-run Check 31)
- [ ] No quality-vs-latency arm mismatch in paired comparisons (re-run Check 32)
- [ ] Every equation discloses known simplifications at the point of use (re-run Check 33)
- [ ] No stale or wrong citations (re-run Check 34)
- [ ] No register shifts or stray punctuation (re-run Check 35)
- [ ] No degenerate table cells (re-run Check 36)
- [ ] Every model name traces to a precise HuggingFace ID (re-run Check 37)

If any item fails, fix and re-audit before reporting completion.

For each check, report:

```
Q[N]: [Question text]
ANSWER: [What was found]
STATUS: [CRITICAL / IMPORTANT / OK]
FIX: [What was done, or "no fix needed"]
```

After all 37 checks, summarize:
- Total issues found (by group)
- Critical issues fixed
- Remaining known limitations (acknowledged in paper)
- Compilation status (errors / overfull / undefined)
- verify_numbers.py status
- Dead macros count
- Derived calculation mismatches count
- Storytelling: setup-payoff closure gaps
- Readability: sentences >40 words, Flesch score range, sections outside 30–50
- Definitions: undefined variables, undefined metrics, missing methodology context
- Forward-use: abbreviations and paper-specific terms used before definition (Check 25)
- Intuition-first: equations without plain-English preamble (Check 26)
- Question-framed: tables without "why" explanations (Check 27)
- Self-contained captions: captions missing pattern or mechanism (Check 28)
- Real-world context: analogies count, dollar translations count
- Readability: `eval_readability.py` pass/fail count, Flesch range, density metrics
- Duplicate content: duplicated paragraphs across sections (Check 30)
- Self-contradiction: threats vs headline claims (Check 31)
- Arm mismatch: quality vs latency model mismatch (Check 32)
- Equation disclosure: undisclosed simplifications (Check 33)
- Citations: stale or wrong model/method citations (Check 34)
- Register: contractions, stray punctuation, tone shifts (Check 35)
- Table cells: degenerate/empty cells (Check 36)
- Model name precision: ambiguous names missing HF ID suffix (Check 37)
- Self-audit: pass/fail per item
- Fix-log: written / not written
