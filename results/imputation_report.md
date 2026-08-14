# Issue #23: Missing-field imputation — results

**Date:** 2026-08-14


## Stratification: random


| K | AR F1(imp) | AR p50 | DM F1(imp) | DM p50 | F1 Δ (DM-AR) | Speed ratio (AR/DM) |
|---|---|---|---|---|---|---|
| 1 | 0.250 | 917 | 0.000 | 823 | -0.250 | 1.11x |
| 2 | 0.591 | 878 | 0.413 | 832 | -0.178 | 1.06x |
| 4 | 0.743 | 903 | 0.301 | 824 | -0.442 | 1.10x |
| 6 | 0.643 | 828 | 0.154 | 829 | -0.489 | 1.00x |
| 8 | 0.688 | 929 | 0.221 | 833 | -0.467 | 1.11x |

## Verdict (P-F predictions)

| ID | Prediction | Result |
|---|---|---|
| P-F1 | DM F1 ≥ AR F1 at all K | **REFUTED** |
| P-F2 | DM latency flat in K (≤20% growth K1→K8) | **PASS** (ratio=1.01) |
| P-F3 | AR latency grows with K | **REFUTED** (ratio=1.01) |
| P-F4 | Cost ratio AR/DM grows with K | **REFUTED** (K=1: 1.11x, K=8: 1.11x) |