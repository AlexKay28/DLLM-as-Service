# EDR-003: First-object-wins as the primary parser policy

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #43, #57 R5

## Hypothesis

When a model over-generates (emits multiple JSON objects in one response),
the parser's choice of which object to score materially changes the measured
F1. The old parser used `text[text.find("{") : text.rfind("}")+1]`, which
captures from the first `{` to the **last** `}` — effectively scoring the
last (most-drifted) copy of each field. This handicaps AR, which rambles
when forced to generate past its natural output length.

## Design

Replace `rfind("}")` with brace-depth tracking: scan from the first `{`,
count nesting depth, and extract the first complete `{...}` block. This is
the "first-object-wins" policy. Also add a "strict" policy (whole output must
be valid JSON) as a secondary column.

Record the parser policy in every result JSON via the provenance helper.

## Prediction

AR-3B at L=512 (forced to generate 512 tokens, ~96% ramble rate) will jump
from ~0.599 F1 (last-wins) to ~0.826 F1 (first-wins) on the identical
generations. DG will be unaffected (it stops at the canvas boundary and
rarely rambles).

## Result

| Model | L | Last-wins F1 | First-wins F1 | Delta |
|---|---|---|---|---|
| AR-3B | 256 | 0.826 | 0.826 | 0 (no ramble at L=256) |
| AR-3B | 512 | 0.599 | 0.826 | **+0.227** |
| DG | 256 | 0.787 | 0.786 | 0 (parser-invariant) |
| DG | 512 | 0.791 | 0.785 | 0 (parser-invariant) |

Prediction confirmed exactly. AR-3B L=512 jumped +0.23 F1 on identical
generations. DG was parser-invariant as expected.

**Artifacts:** `results/issue43/ar3b_L512_preds.jsonl` (raw generations),
`results/issue43_nine_field_f1.json` (re-scored),
`models/baselines/ar_model.py:148-230` (fixed parser)

## Decision

**Accepted:** first-object-wins is the primary parser policy. Strict is
reported as a secondary column. The policy is written into the result JSON,
not assumed.

## Impact

- The §7.1 perturbation result (AR beats DG in 10/10 categories) was scored
  under the old parser. Re-scoring is not needed because perturbation runs
  at L=256 (no ramble), but the policy must be stated.
- Any future experiment that forces `min_tokens > natural_L` must use
  first-object-wins or the result is a parser artifact.
