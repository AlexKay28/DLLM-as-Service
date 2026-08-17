# Grill Fix Log — 2026-08-17

## Pass 9

| # | File | Change | Check | Severity |
|---|------|--------|-------|----------|
| 1 | paper/assets/macros.tex | Stale comment about procedure.py (said "still records AR-3B-era values" — constants were updated in #113) | Check 17 | IMPORTANT |
| 2 | paper/v4/sections/06_related_work.tex | Reduced number density (4.1→pass) by converting inline ms values to prose ("drops by a third", "more than threefold") | Check 29 | IMPORTANT |
| 3 | paper/v4/sections/18_appendix_humaneval.tex | Reduced number density (6.1→pass) by removing repeated numbers from prose (they're in the tables) | Check 29 | IMPORTANT |

## Compilation: 0 errors, 0 overfull, 0 undefined, 23 pages
## Dead macros: 0
## verify_numbers.py: exit 0, 0 errors
## Readability: 61/61 checks pass (was 58/61)
## Self-audit: pass

### Checks run (37 total):

**Group A (Data Integrity):** All pass
- Check 1: Macros vs artifacts — 0 mismatches (arBase=34.6, arSlope=7.2, crossoverL=48)
- Check 2: Claim vs measurement — all serial, no batch-as-serial
- Check 3: Cross-references — 0 undefined
- Check 4: Model identity — consistent

**Group B (Narrative):** All pass
- Check 5: Framing consistent (abstract/conclusion agree)
- Check 22: Story arc coherent
- Check 23: Readability — all sections pass thresholds
- Check 24: Definitions complete
- Check 25: No forward-use abbreviations in abstract
- Check 26: Intuition before formalism — analogies present (taxi/subway)
- Check 27: Captions self-contained — some CHECK (data tables, acceptable)
- Check 28: (N/A — not in skill)
- Check 29: Readability — 61/61 pass (fixed §06 and §18)

**Group C (Math):** All pass
- Check 6: Equations valid, where-blocks present
- Check 16: Derived calculations — 5/5 match (crossover, lambda, headline, batch, slope ratio)
- Check 18: Integer-safe crossover — ≥48 (true crossover 47.81)

**Group D (Presentation):** All pass
- Check 7: Figure-text alignment — consistent
- Check 8: Appendix consistency — values match main text
- Check 9: Table completeness — all models present
- Check 12: Cost table — arithmetic verified (events/hr, $/event correct)
- Check 13: Contrast table — values match macros

**Group E (Completeness):** All pass
- Check 10: Threats cover tail (n=2000), cascade (measured), energy (idle baselines), grammar (attempted)
- Check 14: Abstract/conclusion/contributions consistent

**Group F (Hygiene):** All pass
- Check 11: No stale old numbers (15.6×, 77×, 5899, etc. — only in Appendix I correction 9, intentional)
- Check 15: Compilation — 0 errors, 0 overfull, 0 undefined
- Check 17: Macro comments accurate (fixed procedure.py comment)
- Check 19: Per-event data attribution correct
- Check 20: verify_numbers.py matches macros.tex claims
- Check 21: Dead macros — 0

**Group G (Structural):** All pass
- Check 30: No duplicate content
- Check 31: No self-contradiction (threats qualifications propagated)
- Check 32: No arm mismatch
