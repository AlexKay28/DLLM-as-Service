# Issue #27: Perturbation robustness

## Clean baseline

- AR-3B F1: 0.8268
- DM F1: 0.7390
- AR advantage: 0.0878

## At intensity i=1.0

| Category | AR F1 | DM F1 | AR drop | DM drop | DM/AR ratio |
|----------|-------|-------|---------|---------|-------------|
| whitespace | 0.827 | 0.658 | -0.000 | 0.081 | inf (AR flat) |
| field_reorder | 0.782 | 0.615 | 0.045 | 0.124 | 2.8x |
| separator_change | 0.815 | 0.646 | 0.011 | 0.093 | 8.2x |
| abbreviation | 0.829 | 0.734 | -0.002 | 0.005 | inf (AR flat) |
| timestamp_format | 0.746 | 0.635 | 0.080 | 0.104 | 1.3x |
| missing_field | 0.809 | 0.672 | 0.018 | 0.067 | 3.7x |
| case_variation | 0.714 | 0.608 | 0.113 | 0.132 | 1.2x |
| wrapping_noise | 0.817 | 0.698 | 0.010 | 0.041 | 4.1x |
| synonym_substitution | 0.807 | 0.714 | 0.020 | 0.025 | 1.3x |
| encoding_noise | 0.821 | 0.704 | 0.006 | 0.035 | 5.8x |

**DM worse in 10/10 categories** — hypothesis refuted.
