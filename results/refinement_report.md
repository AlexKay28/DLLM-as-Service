# Issue #24: Iterative refinement loops

| Model | Round 1 F1 | Round 2 F1 | Round 3 F1 | Gain |
|-------|-----------|-----------|-----------|------|
| AR-3B | 0.822 | 0.821 | 0.821 | -0.001 |
| DM-8B | 0.747 | 0.763 | 0.770 | +0.023 |

## Verdicts

- **P-F5** (DM F1 gain > 0.01): PASS (gain=+0.023)
- **P-F6** (AR F1 flat): PASS (gain=-0.001)
- **P-F9** (DM R3 > AR R1): REFUTED (DM=0.770, AR=0.822)
