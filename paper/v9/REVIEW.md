# V9 editorial and evidence review

## Main story

The seven-page main text focuses on one similarly sized pair: Qwen3-30B-A3B and DiffusionGemma-26B-A4B. Its sequence is motivation and takeaways → output-length intuition → direct quality and generation-budget results → batching and cost → suitable workloads → limits and conclusion.

Figure 1 remains directly below the abstract. Numerical comparisons use tables. The body separates a long response from a sequence of dependent calls, and explains how generation latency consumes part of an end-to-end deadline. The smaller AR-3B chain and wide-schema studies, detailed serving controls, energy estimates, and protocol detail remain in appendices; they are not evidence of AR-30B investigation performance.

## Direct quality comparison

The saved AR seed-42 predictions and the first 300 DG repeat-1 predictions can be compared against the same prepared reference prefix, following the respective runners' input order. V9 freezes those parser-derived references and reuses the repository's scorer.

- Nine-field F1: AR-30B **0.8698**, DG **0.8518**.
- Schema validity: AR-30B **99.33%**, DG **99.00%**.
- Mean paired difference, DG minus AR: **−0.0181**; paired percentile-bootstrap 95% interval **[−0.0289, −0.0086]**.

These close averages support a small DG deficit, not a proof of equal quality. No equivalence margin is selected after seeing the result. Pairing follows the saved-output runners; shared templates, reference construction, and a single event prefix constrain generalization.

The 300-event paired quality table includes its own latency medians. It is explicitly separate from the 900-AR/2,000-DG timing comparison used in the abstract and cumulative plot.

## Generation budgets and the plot

The old blue vertical band marked uncertainty in the AR median; it did not represent another set of responses. The updated plot preserves the empirical cumulative curves, uses percentages, removes the bands, and marks the 1,000 ms threshold.

At 1,000 ms, 1,999/2,000 DG generations (99.95%) and 694/900 AR generations (77.11%) have completed. At 500 ms, the fractions are 99.8% and 0%. A separate table also applies schema validity, preventing completion from being mistaken for usable or correct output. Semantic correctness and downstream time are not measured by these counts.

## Provenance correction

The DG anchor summary records vLLM 0.25.1, whereas the DG n=2,000 tail-summary file records 0.11.0 for the same timing summary. This revision preserves both artifacts, documents the conflict, and removes the assertion of exact stack parity. The latency result describes the saved runs; it does not isolate an architecture-only effect.

## Reading and evidence checklist

- [x] Main text reaches the conclusion within eight pages, retaining 11-point type and original margins.
- [x] First-page figure, bold takeaways, and the decoding/deadline schematics remain.
- [x] Primary model identities are explicit; smaller baselines are supporting evidence only.
- [x] Direct paired quality, schema validity, and timing appear together with sample sizes.
- [x] Generation-budget completion is distinct from correctness and end-to-end deadlines.
- [x] Cost coefficients have names, meanings, and units before numerical substitution.
- [x] Longer-output intuition is not presented as a measured universal advantage.
- [x] Quality and timing reanalysis is reproducible from frozen references and saved outputs.
- [x] V8 and original results remain unchanged.
