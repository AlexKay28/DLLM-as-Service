# Editorial review: v8 → v9

V9 responds to reading feedback while preserving the existing measurements and scientific boundaries. No model inference, experiment changes, or new deployment evidence are introduced. V8 is unchanged.

## Changes

- Shortened the abstract to a standalone account of motivation, comparison, result, and deployment implications.
- Added three bold takeaways before the experimental details: measured serial latency, potential deadline value, and the quality/concurrency tradeoff.
- Added a roadmap following the actual section sequence and grouped the existing protocol details in an experimental-setup subsection.
- Defined the main acronyms, model sizes, decoding terms, and metrics at first use. The leading cost figure appears immediately below the abstract, with abbreviations explained in its caption.
- Added a structured-log deadline example distinguishing completion, validity, quality, and timeliness, including queueing and downstream work.
- Explained why batching throughput and deadline compliance answer different questions; aligned the conclusion with the opening takeaways.

## Storyline checklist

- [x] The abstract explains the task and expands BlueGene/L before BGL.
- [x] The introduction explains why a late record can fail a timely decision.
- [x] Three highlighted takeaways appear before detailed protocol discussion.
- [x] The roadmap matches setup → cost models → measurements → deployment constraints → workload implications.
- [x] The primary AR-30B and secondary dense AR-3B comparisons remain explicit.
- [x] The 2.4× result is a measured median-latency comparison, not a quality-matched speedup claim.
- [x] Long-output projections remain conditional; output budgets are not substituted for generated lengths.
- [x] The deadline example is illustrative, not a measured end-to-end success rate.
- [x] Complete output still requires task-appropriate validity and quality.
- [x] Throughput, individual response time, and utilization-dependent cost remain distinct.
- [x] The conclusion returns to the same three takeaways.

## Follow-up reading feedback

- Restored Figure 1 as the first-page visual summary and kept the introduction and takeaways together on the following page.
- Removed the end-to-end measurement disclaimer from the abstract; the main text still explains what the saved generation timings cover. No extrapolation is presented as measured deadline success.
- Replaced Figure 5's two panels with one ratio plot: median and exploratory p99.9 ratios remain distinct, and median points carry absolute cumulative AR/DG times. Values come from `results/chain_analysis.json`; the builder checks that reconstructed median ratios agree with its saved ratios.
- Added two unnumbered conceptual diagrams for decoding schedules and the full deadline path. These explain the workflow without implying new measurements, and preserve the numbered figures' identities.
- Expressed equations 1–3 with named coefficients and units before substituting measured constants in the crossover calculation.
- Simplified the cost-model explanation, deadline discussion, and conclusion while preserving the scientific scope.

## Evidence retained after revision

The limitations identified in [v8's evidence review](../v8/REVIEW.md) remain: quality matching for AR-30B, natural long-output validation, HumanEval protocol reconciliation, optimized serving controls, and controlled deadline/energy/cost evaluations. V9 improves how these results are explained; it does not resolve those experimental questions.
