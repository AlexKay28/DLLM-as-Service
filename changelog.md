# Changelog

## 2026-08-10 — Issue #2: Datasets & Sources pipeline

Built the data pipeline that turns raw LOGHUB corpora into FlowFill's 12-field
schema splits, unblocking the downstream validation issues (#3–#8).

- **`datasets/download.py`** — resumable downloader (pure stdlib) for HDFS/BGL/
  Thunderbird (Zenodo 8196385) + Spirit (USENIX HPC4). Verifies against the
  server's Content-Length; supports `--only`, `--extract`, `--verify`,
  `--samples` (2k LOGHUB samples for smoke tests).
- **`datasets/prepare.py`** — per corpus: reservoir-sample 600k lines → temporal
  split (last 200k = test) → parse with Drain (`logparser3`, regex fallback) →
  emit 12-field schema JSONL + `manifest.json`. Bounded memory (works on 211M-
  line Thunderbird). `--smoke` mode for CI.
- **`datasets/README.md`** — documented both scripts + the actual directory
  layout.
- **`.gitignore`** — excludes `datasets/raw/`, `datasets/parsed/`,
  `datasets/splits/`.

**Verified:** smoke test on the 2k samples produces 100% schema-valid records
across HDFS/BGL/Thunderbird; `run_experiments.py` consumes the splits directly.

**Still open:** semantic-field labels (`event_type`, `affected_entity`,
`remediation_hint`) are placeholders pending a teacher LLM (tracked by #6).

## 2026-08-11 — Issue #8: Missing-Baselines Gap (Track 1)

Ran the rule-based + Drain-only baselines on real LOGHUB BGL data (200k test
events) across all 5 scenarios, producing the first real numbers that
contextualize FlowFill's claims against classical methods.

- **`datasets/prepare.py`** — fixed BGL parser: the real `BGL.log` has 9
  leading fields (`<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <RAS>
  <Component> <Level> <Content>`), not 7 as assumed. Now extracts correct
  component/severity/host. Also writes raw test lines (`*.raw`) for the
  experiment runner.
- **`models/baselines/drain_only.py`** — same BGL 9-field fix in the fallback
  parser; initialized `label` to avoid UnboundLocalError on HDFS.
- **`models/experiments/run_baselines_bgl.py`** — new runner that executes all
  5 scenarios (clean, drift, perturbation, cross_source, semantic) on real BGL
  data with tqdm progress bars. Full 200k for clean/drift/cross_source; 20k
  subsample for the 132-run perturbation sweep.
- **`results/baselines_scenarios.json`** — 154 experiment results across all
  scenarios.
- **`results/baselines_report.md`** — analysis answering the issue's central
  question.

**Key findings:**
- Drain-only achieves **0.833 Field-F1** vs FlowFill's 0.907 on clean BGL —
  the gap is entirely from semantic fields Drain cannot produce.
- A **hybrid (Drain + small LLM)** would achieve ~**0.865 F1 at ~11.7x
  throughput** (50,768 vs 4,350 ev/s) — small enough to qualify the paper's
  end-to-end diffusion claim on stable logs.
- Classical baselines **match FlowFill on anomaly detection** (1.000 vs 0.865
  Anomaly-F1) using native BGL labels.
- Under mixed perturbation (intensity=1.0), rule-based collapses to 0.426 F1
  and Drain to 0.623 — this is where end-to-end LLMs have a genuine advantage.

**Not done:** DeepLog, LogBERT, and a real hybrid (Drain + classifier) are
higher-effort baselines not yet implemented (tracked in the issue).
