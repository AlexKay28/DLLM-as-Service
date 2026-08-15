# EDR-012: The five defects and their asymmetry

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #65, #63

## Hypothesis

Comparison harnesses written by someone hoping the new architecture wins
will fail asymmetrically — the defects cluster on the arm the experimenter
was not rooting for. In this repo, that arm is AR. All five defects were
found by re-reading committed artifacts, not by running new experiments.

## Design

Catalogue every defect found, its mechanism, its magnitude, and which arm
it biased against. Write CHECKLIST.md as a standalone, citable artifact.
Extend `paper/verify_numbers.py` to assert the five magnitudes.

## Prediction

Four of five will have biased against AR. The fifth (provenance) will be
neutral — a correctness-of-record defect, not a bias.

## Result

| # | Defect | Magnitude | Biased against | Found by |
|---|---|---|---|---|
| 1 | Decoding config | F1 0.779→0.844 | AR | re-reading generation_config.json |
| 2 | Serving parity | p50 1841→413 ms (4.5×) | AR | naming the serving stack |
| 3 | Parser policy | F1 0.599→0.826 (+0.23) | AR | re-scoring saved generations |
| 4 | Metric defaults | completeness 0%→100% | AR | feeding synthetic inputs |
| 5 | Provenance | 3 files mislabelled | neutral | grepping the runner's write path |

Prediction confirmed: 4 of 5 biased against AR. All found by re-reading
artifacts.

**Artifacts:** `CHECKLIST.md`, `paper/verify_numbers.py` (all 5 magnitudes
asserted and passing)

## Decision

**Accepted:** the asymmetry is the finding. The paper's §2 reports it as a
methodology section (not an appendix, per EDR-009 Q5 resolution). The
mechanism for each bias is stated: the AR arm inherits defaults, naive
serving, and the production-grade infrastructure of the baseline it is
trying to beat.

## Impact

- C13 (4 of 5 biased) and C14 (all found by re-reading) go green.
- The checklist is a secondary contribution — it supports the systems-paper
  framing but does not compete with the cost laws for the headline.
- Future experiments in this repo inherit the provenance helper, 9-field
  scoring, and parser policy as defaults.
