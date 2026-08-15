# Experimental Decision Records (EDR)

This directory records research decisions — hypotheses tested, experimental
designs chosen, predictions made, and what the data said. Each EDR is a
permanent record of *why we decided what we decided*, written at the point of
decision.

## How EDRs differ from ADRs

| | ADR (Architecture Decision Record) | EDR (Experimental Decision Record) |
|---|---|---|
| Domain | Software architecture | Research methodology and claims |
| Question | "How do we build this?" | "What do we believe is true, and did the data confirm it?" |
| Reversibility | Decisions can be superseded | Hypotheses can be refuted (not reversed — the refutation stands) |
| Structure | Context / Decision / Consequences | Hypothesis / Design / Prediction / Result / Decision |
| Tone | Engineering rationale | Pre-registered prediction + measured outcome |

## EDR format

Each EDR has these sections:

1. **Hypothesis** — what we believed before running the experiment
2. **Design** — how we tested it (enough detail to reproduce)
3. **Prediction** — what we expected to see, stated *before* looking at data
4. **Result** — what actually happened, with the artifact path
5. **Decision** — confirm / refute / re-scope / retitle, and why
6. **Impact** — what this changes in the paper, the decision procedure, or the
   next experiment

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [EDR-001](001-pin-the-stack.md) | Pin the measurement stack to vLLM 0.25.1 | Accepted | 2026-08-15 |
| [EDR-002](002-nine-field-scoring.md) | Drop 3 degenerate semantic fields from the headline metric | Accepted | 2026-08-15 |
| [EDR-003](003-parser-policy.md) | First-object-wins as the primary parser policy | Accepted | 2026-08-15 |
| [EDR-004](004-retitle-batching-mechanism.md) | State the batching cap as correlation, not mechanism | Accepted | 2026-08-15 |
| [EDR-005](005-two-variable-cost-law.md) | T_DG is a function of input AND output length | Accepted | 2026-08-15 |
| [EDR-006](006-canvas-size-tuning.md) | Canvas=128 is a free improvement; canvas=64 is not viable | Accepted | 2026-08-15 |
| [EDR-007](007-quality-matched-pair.md) | AR-1.5B is the quality-matched baseline, not AR-7B | Accepted | 2026-08-15 |
| [EDR-008](008-dg-base-decomposition.md) | 93% of DG's base is prefill+fixed, not denoising | Accepted | 2026-08-15 |
| [EDR-009](009-paper-identity.md) | Paper identity — resolving the interrogation tensions | Accepted | 2026-08-15 |
| [EDR-010](010-the-fair-comparison.md) | The fair comparison — 3 DG wins + 3 AR wins | Accepted | 2026-08-15 |
| [EDR-011](011-the-crossover.md) | L*≈59 tokens, λ*≈2.6 ev/s — derived from the laws | Accepted | 2026-08-15 |
| [EDR-012](012-five-defects.md) | Five defects, four biased against AR, all found by re-reading | Accepted | 2026-08-15 |
| [EDR-013](013-decision-procedure.md) | Decision procedure — implemented, tested, CLI | Accepted | 2026-08-15 |
| [EDR-014](014-canvas-config-knob.md) | Canvas is tunable but not a design lever | Accepted | 2026-08-15 |
| [EDR-015](015-negative-results.md) | Five refuted + three withdrawn hypotheses | Accepted | 2026-08-15 |
| [EDR-016](016-generalization-hdfs.md) | Cost-law form generalises to HDFS | Accepted | 2026-08-15 |
| [EDR-017](017-llada-generalization.md) | LLaDA under dInfer — form does NOT clearly transfer | Accepted | 2026-08-15 |
| [EDR-018](018-energy-measurement.md) | DG uses 4.3× less energy per event at compute-matched pair | Accepted | 2026-08-15 |
| [EDR-019](019-memory-probe.md) | Long-context memory probe — inconclusive (measurement method wrong) | Inconclusive | 2026-08-15 |
| [EDR-020](020-infilling-experiment.md) | Infilling advantage refuted — AR trivially regenerates suffix | Accepted | 2026-08-15 |
| [EDR-021](021-real-life-applications.md) | Real-life applications where DM is the right choice | Proposed | 2026-08-15 |
| [EDR-022](022-adaptive-k-and-graceful-degradation.md) | Adaptive K and graceful degradation as DM structural properties | Proposed | 2026-08-15 |

## Rules

- **Write EDRs at the point of decision**, not retroactively. If you find
  yourself writing "we always believed...", stop and write what you actually
  believed before the experiment.
- **State the prediction before the result.** This is the load-bearing
  distinction from a post-hoc narrative. If you didn't predict, say so.
- **Refuted hypotheses stay in the index.** They are not failures; they are
  the findings. Deleting a refuted EDR is the research equivalent of p-hacking.
- **One decision per EDR.** If an experiment produced two decisions, write two
  EDRs and cross-reference them.
- **Every EDR links to its artifact.** A decision without a committed artifact
  is an opinion.
