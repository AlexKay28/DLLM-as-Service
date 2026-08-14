# Issue #25: Bounded-deadline completion

## AR-3B (vLLM, max_tokens = D/6.28)

| D(ms) | F1 | Field% | Record% | p50ms |
|-------|----|--------|---------|-------|
| 200 | 0.315 | 0.0% | 0.0% | 203 |
| 400 | 0.315 | 0.0% | 0.0% | 403 |
| 600 | 0.315 | 0.0% | 0.0% | 605 |
| 800 | 0.621 | 61.5% | 61.5% | 805 |
| 1000 | 0.814 | 97.5% | 97.5% | 808 |
| 1500 | 0.827 | 100.0% | 100.0% | 807 |
| 2000 | 0.827 | 100.0% | 100.0% | 807 |

## DM (LLaDA-8B, K=8, one canvas)

- F1: 0.732
- Field completeness: 100.0%
- Record completeness: 100.0%
- p50: 1077ms

## Verdicts

- **P-F7** (DM record completeness > AR at D=1500): REFUTED (DM=100.0%, AR=100.0%)
- **P-F8** (AR F1 collapses below D=1000): PASS (D=200 F1=0.315, D=1000 F1=0.814)
