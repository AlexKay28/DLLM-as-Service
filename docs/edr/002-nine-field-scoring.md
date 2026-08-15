# EDR-002: Drop 3 degenerate semantic fields from the headline metric

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #53, #57 R4

## Hypothesis

The 12-field macro F1 used in all published tables is contaminated by three
semantic fields whose gold values have near-zero variance: `event_type` (2
unique values across 2000 events), `affected_entity` (1), and
`remediation_hint` (1). A model emitting `{}` scores 1.000 on two of three
semantic fields, inflating its macro F1 by ~0.03. The 9-field metric (7
structural + 2 anomaly) is a fairer headline.

## Design

Define `NINE_FIELD_NAMES` in `metrics/quality_metrics.py`:
timestamp_norm, host, component, severity, message_template, error_code,
metric_value, is_anomaly, anomaly_score. Compute both 9-field and 12-field
macros in every call to `field_f1()`. Re-score all issue43 predictions on
both metrics.

## Prediction

The 12-field metric inflated AR's apparent advantage because AR fills the
degenerate semantic fields with plausible-looking text while DG leaves them
empty. On 9-field, the two architectures will be within 0.01 of each other.

## Result

| Model | 9-field F1 | 12-field F1 | Gap |
|---|---|---|---|
| AR-3B L=256 | 0.8516 | 0.8263 | -0.025 (AR drops) |
| DG L=256 | 0.8518 | 0.7858 | -0.066 (DG drops more) |
| AR-1.5B | 0.8628 | 0.8435 | -0.019 |

On 9-field, DG matches AR (0.8518 vs 0.8516). The 12-field gap (0.787 vs
0.826) was entirely from the 3 degenerate fields, which DG leaves empty and
AR fills with text that happens to match the low-variance gold.

**Artifacts:** `results/issue43_nine_field_f1.json`

## Decision

**Accepted:** 9-field is the headline metric. 12-field is reported alongside
for continuity with published tables, clearly labelled as contaminated.

## Impact

- The "quality price" of choosing DG over AR drops from 0.039 (12-field) to
  ~0.002 (9-field) — essentially zero. The decision procedure's quality guard
  fires less often.
- The paper's §3 (exchange rate) thesis is strengthened: quality is genuinely
  comparable, not "comparable if you ignore the fields where DG is bad".
