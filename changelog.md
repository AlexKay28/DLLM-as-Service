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
