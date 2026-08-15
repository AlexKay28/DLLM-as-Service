# EDR-016: Cost-law form generalises to a second corpus

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #67, #63

## Hypothesis

The staircase cost shape (flat within a canvas, step at boundaries) is a
property of block-structured decoding, not a property of the BGL corpus. If
it holds on HDFS, the form transfers while only the constants move. The step
cost should be ~120 ms on both corpora (it is a model property, not a data
property).

## Design

Run DiffusionGemma on HDFS corpus, L ∈ {128, 256, 512, 2048}, n=200/cell,
K=8, pinned stack. Re-run AR-3B on HDFS on the same stack for paired
comparison. Fit base and step on HDFS, compare to BGL.

## Prediction

- HDFS step ≈ 120 ms (same as BGL — model property)
- HDFS base may differ (HDFS prompts are shorter → lower prefill)
- AR HDFS p50 ≈ BGL p50 (already shown in #50: +0.2%)

## Result

| L | BGL p50 (ms) | HDFS p50 (ms) | ratio |
|---|---|---|---|
| 128 | 378.3 | 366.7 | 0.97× |
| 256 | 378.3 | 341.4 | 0.90× |
| 512 | 500.9 | 469.1 | 0.94× |
| 2048 | 1212.0 | 1192.2 | 0.98× |

Step costs:
- BGL: 122.6 ms (L=256→512), 118.5 ms/canvas (L=512→2048)
- HDFS: 127.7 ms (L=256→512), 120.5 ms/canvas (L=512→2048)

Step is preserved: ~120 ms on both corpora. Base is lower on HDFS (shorter
prompts → less prefill), consistent with EDR-005 (T_DG depends on L_in).

AR-3B HDFS: L=256 p50=1616.5 ms (BGL: 1616.0 ms, +0.03%), L=512 p50=3223.8
ms (BGL: 3225.1 ms, -0.04%). AR law is corpus-independent.

**Artifacts:** `results/issue67_dg_hdfs_*.json`,
`results/issue67_ar3b_hdfs_*.json`

## Decision

**Accepted:** the cost-law FORM generalises to HDFS. The staircase holds
(step ~120 ms on both corpora). The constants move (base is lower on HDFS
due to shorter prompts), exactly as the two-variable law predicts. AR law
is corpus-independent.

The word "law" is licensed for the form. The constants are corpus-conditional.

## Impact

- C18 and C19 go green.
- §9 threat 6 ("single corpus BGL") is narrowed: the form holds on two
  corpora; the constants are reported per-corpus.
- The paper states: "the cost shape is a property of block-structured
  decoding, verified on two LOGHUB corpora. Constants (base, prefill slope)
  depend on the corpus through prompt length; the step is a model property
  (~120 ms, corpus-independent)."
- Arm (c) — LLaDA under dInfer — is still queued (GPU pending). If it
  confirms the staircase at a different block_length, the form generalises
  across model families too. If not, it is DiffusionGemma-specific.
