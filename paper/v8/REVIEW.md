# Editorial and evidence review: v7 → v8

The measured serial-latency advantage remains a useful result. V8 presents it as a scoped systems measurement, with a clearer distinction between latency, throughput, output quality, and architectural explanation. This review found substantive contradictions as well as writing issues; the changes therefore go beyond copyediting.

The source is v7 at `d4ffeac`. V7 and all earlier versions remain unchanged. No inference was rerun and no original result artifact was edited.

## Material corrections

| Area | V7 issue | V8 treatment and evidence |
| --- | --- | --- |
| Main latency result | Alternated between an AR cell **mean** of 945 ms and a repeated **median** of 924 ms, usually calling both a 2.5× result. | Uses median 924.4 ms versus 379.7 ms, approximately **2.4×**, for the repeated primary pair. Retains the original mean-versus-model table, explicitly labeled. Sources: `results/anchors/ar30b_stats.json`, `results/issue111_qwen3_serial_sweep.json`. |
| Sample size and significance | Claimed 3,200 DG events in a comparison computed with 2,000; interpreted zero Monte Carlo exceedances as an arbitrarily small p-value. | Uses the actual 2,000 DG and 900 AR events. Describes zero exceedances in 100,000 permutations and the finite Monte Carlo estimate near 10⁻⁵, with exchangeability and run-level limitations. No new significance test is claimed. |
| Tail latency | Claimed AR had lower tail latency because its max/median ratio was smaller. Mixed dense AR-3B statistics into an AR-30B table. | Uses the primary pair throughout the summary: DG p99 **490.7 ms**, AR-30B **1400.4 ms**. Separates absolute latency from relative variability. A sample maximum is not a worst-case guarantee. |
| Output-length claim | Said DG was permanently faster above 48 tokens, while another table reversed the comparison at larger budgets. | Separates actual generated length, configured budget, and decoded canvas count. Identifies the 48-token intersection as a conditional fitted-cost crossover, not a valid-record threshold or universal speedup. |
| Cost-model evidence | Treated five budget cells as broad output-length validation; three cells actually stopped at the same EOS length. | States the three distinct actual lengths, short support range, AR mean/DG median distinction, and long-output extrapolation. Redraws the leading figure with a dashed extrapolation and a separate observed-median panel. |
| Cache mechanism | Asserted DG could not cache its prompt and paid prompt prefill K times; used this to explain a fundamental batching ceiling. | Removes the causal claim. Google's model card describes an encoder caching context and a decoder attending to it. Canvas-state recomputation is distinguished from context caching. The observed engine-limited batching gains remain. |
| Base decomposition | Presented a two-point K interpolation as physical identification of a 62% removable prefill cost. | Explains that the split is algebraic, not a profile or validated zero-step floor. `issue70_profiler_attribution.json` itself labels the method analytical. |
| Energy | Used 118.7 J/event and a 28.3× ratio, inconsistent with the current serial artifact. Also claimed DG was busy longer. | Corrects AR to **166.5 J/event**, versus 4.2 J for DG in a different run; descriptive ratio approximately 40×. Reports mismatched baselines and power-sampling limitations. Checks power × duration / event count. |
| Deadline / completeness | Claimed a fully decoded canvas guarantees valid JSON, safe preemption, and a capability AR cannot possess. | Distinguishes token completion from syntax/schema validity, selecting K in advance from interruption, and adaptive effort from a diffusion-exclusive property. Retains observed K tradeoffs without guarantees. |
| Applications and cost | Recommended driving/trading without experiments; projected linear speedup from sixteen weight-only replicas; treated busy-time pricing as annual savings on an always-on GPU. | Removes deployment endorsements and unmeasured co-residency speedup. Replaces the rental-price table with an explicit utilization-dependent cost expression. |
| Labels and metrics | Described anomaly references as native tags alone and schema validity as key presence. | Documents the actual severity-or-tag labeling rule, seven structural fields, empty-value F1 behavior, and the harness's required-field/type/enum validation. Source: `datasets/prepare.py`, `metrics/quality_metrics.py`, `metrics/schema.py`. |
| HumanEval | Claimed quality-independent cost-shape generalization and deadline success despite DG's 31.1% syntax validity and latency exceeding the nominal deadline. Misread first-round versus total-chain medians as per-hop growth. | Retains the tables as exploratory evidence, discloses lower pass@1, identifies the token-budget proxy, and flags the 0.287 versus 0.854 first-round discrepancy. Clarifies that chain totals sum generation time and omit test/orchestration time. |
| Citations and disclosures | Cited Qwen2.5 for Qwen3; the Hoefler citation pointed to an unrelated identity-management paper; claimed no LLM editing. | Adds the Qwen3 report and correct Hoefler–Belli paper, corrects the DG model-card entry, and discloses assisted editing. Moves venue checklist self-certification out of the scientific paper. |
| Tone | Repeated taxi/subway metaphors, broad “right choice” assertions, and speculation about experimenters' motives. | Uses direct descriptions of measurements and distinguishes our harness corrections from allegations about other researchers. |

All nine figure panels/sets used in the revision were regenerated from the corresponding committed measurements or explicitly labeled cost-model constants. The new figures remove claims such as “DG pays for the prompt K times,” “diffusion ceiling,” and “two orders of magnitude apart.” The old batching figure also drew a diffusion reference line by multiplying a throughput multiplier by an arbitrary 0.5 events/s; v8 removes that line.

## Evidence to resolve before making stronger claims

1. **Quality matching for the main pair.** The DG/dense-AR-3B equivalence interval does not establish equivalence for AR-30B. A paired main-pair evaluation, justified margin, validity analysis, and per-field breakdown would support a quality-matched speedup claim. The current text makes only a conditional quality statement.
2. **HumanEval protocol reconciliation.** The DG first-round pass fraction is 0.2866 in `issue115_humaneval/dg_K8.json` and 0.8537 in `issue117_chain/dg_K8.json`. The latter lacks generated code for an output-level audit. Do not treat these as replications or use them to support the headline until the prompts, revisions, parser, and execution protocol are reconciled. Both evaluators also use a `PASS` substring in stdout as their success signal; a robust follow-up should verify the subprocess exit status and isolate trusted test success from generated-code output.
3. **Naturally long outputs.** The multi-record table is a fixed-prompt projection. A real task would change input length, generated content, and possibly quality. It does not validate the extrapolated AR slope or larger ratios.
4. **Mechanism attribution.** A cache-enabled optimized diffusion control and a real hardware profile are needed before claiming a fundamental batching ceiling or identifying shares of prefill/denoising cost.
5. **Controlled secondary comparisons.** Deadline claims need actual on-time correct-output rates; energy needs comparable power sampling and operating conditions; deployed cost needs utilization and concurrency measurements. These are prerequisites for those stronger claims, not prerequisites for reporting the scoped serial measurements.

## Sources checked externally

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388): correct reference for the main AR model.
- [Google DiffusionGemma model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it): documents cached context, encoder–decoder design, and general multimodal scope. Accessed 7 September 2026. This is current model documentation, not proof of every historical runtime's implementation.
- [Hoefler and Belli, Scientific Benchmarking of Parallel Computing Systems](https://htor.inf.ethz.ch/publications/index.php?pub=222): replaces the unrelated arXiv identifier in v7.
- [XGrammar](https://arxiv.org/abs/2411.15100) and [EAGLE](https://arxiv.org/abs/2401.15077): checked the relevant primary references.

This was a targeted source check, not a systematic novelty search or a full bibliography audit.

## Validation

Validation results are recorded in `VALIDATION.md`. The paper is substantially more defensible as written, but the checks do not certify publication readiness for a particular venue. Choose the venue, adapt its current template and disclosure requirements, and have the author review the scientific scope before submission.
