# Grill Fix Log — 2026-08-16

## Pass 4 (evolved 24-check skill)

| # | File | Change | Check | Severity |
|---|------|--------|-------|----------|
| 1 | sections/07_threats.tex | Added energy idle-baseline limitation paragraph (AR-30B 92.7W, AR-1.5B 58.6W, DG 52.6W; DG not re-measured at AR-30B idle) | Check 10 | IMPORTANT |
| 2 | sections/16_appendix_forced.tex | Split 77-word sentence into two (cost-law fitting + EOS explanation) | Check 23 | IMPORTANT |

## False positives skipped (from catalog)

| Pattern | Check | Why skipped |
|---------|-------|-------------|
| dgStep compared against L256 p50 (378.3 vs 119.5) | Check 1 | dgStep is canvas-step median, not base p50; separate verification confirmed 119.5 |
| DG events/hr 9499 vs 9504 | Check 12 | Table uses dgBase=378.8 (9504), my script used 379 (9499); 0.05% difference |
| AR-1.5B cost $0.000854 vs $0.0009 | Check 12 | Rounding to 4 decimal places |
| N=1 L=123 AR=5946 vs table 5900 | Check 8 | First row is measured p50 (5900), not cost-law extrapolation (5946) |
| 2 equations "without where-blocks" | Check 24 | align inherits variables from preceding equation's where-block |
| 02_cost_laws 94-word "sentence" | Check 23 | Equation where-block parsed as prose by regex |

## Compilation: 0 errors, 0 overfull, 0 undefined, 20 pages
## Dead macros removed: 0 (none found)
## Self-audit: see below
