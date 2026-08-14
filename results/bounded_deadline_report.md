# Issue #25: Bounded-deadline completion

## AR-3B (vLLM, max_tokens = D/6.28)

| D(ms) | F1 | Field% | Record% | p50ms |
|-------|----|--------|---------|-------|
| 200 | 0.319 | 0.0% | 0.0% | 203 |
| 400 | 0.319 | 0.0% | 0.0% | 404 |
| 600 | 0.319 | 0.0% | 0.0% | 604 |
| 800 | 0.613 | 36.9% | 0.0% | 805 |
| 1000 | 0.808 | 60.7% | 0.0% | 808 |
| 1500 | 0.821 | 62.7% | 0.0% | 808 |
| 2000 | 0.821 | 62.7% | 0.0% | 807 |

## DM (LLaDA-8B, K=8, one canvas)

- F1: 0.732
- Field completeness: 100.0%
- Record completeness: 100.0%
- p50: 1077ms

## Verdicts

- **P-F7** (DM record completeness > AR at D=1500): PASS (DM=100.0%, AR=0.0%)
- **P-F8** (AR F1 collapses below D=1000): PASS (D=200 F1=0.319, D=1000 F1=0.808)
