# Datasets

Datasets used in the FlowFill paper: **"Masked-Diffusion Language Models as a Bounded-Latency Monitoring Engine for Streaming Logs"**.

All four corpora come from **LOGHUB** and the **USENIX CFDR HPC4** collection. They are publicly available for research.

---

## Overview

| Corpus | Source | Lines (raw) | Sampled | Test Anomaly % | Role |
|---|---|---|---|---|---|
| HDFS | LOGHUB (Zenodo) | 11,175,629 | 600k | 2.9 | Distributed FS logs |
| BGL | LOGHUB (Zenodo) | 4,747,963 | 600k | 7.3 | Blue Gene/L supercomputer |
| Thunderbird | LOGHUB (Zenodo) | 211,212,192 | 600k | 4.1 | Thunderbird supercomputer |
| Spirit | USENIX CFDR | ~272M | 600k | 5.8 | Spirit supercomputer |
| **Total** | — | — | **2.4M** | **5.0** | Temporal split: last 200k/corpus held out |

**Split strategy:** Most recent 200k lines per corpus held out as a temporal test split to avoid leakage.

---

## Download Links

### HDFS (Hadoop Distributed File System)

- **Lines:** 11,175,629 (1.47 GiB)
- **Time span:** 38.7 hours
- **Labeled:** Yes (anomaly labels via handcrafted rules, sliced by block IDs)
- **Download:**
  ```bash
  wget https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1 -O HDFS.zip
  unzip HDFS.zip -d HDFS/
  ```
- **LOGHUB page:** https://github.com/logpai/loghub/tree/master/HDFS
- **Original paper:** Wei Xu et al., "Detecting Large-Scale System Problems by Mining Console Logs" (SOSP'09)

### BGL (Blue Gene/L)

- **Lines:** 4,747,963 (708.76 MiB)
- **Time span:** 214.7 days
- **Labeled:** Yes (alert category tags — first column "-" = non-alert, other = alert)
- **Source:** BlueGene/L supercomputer at Lawrence Livermore National Labs (LLNL), 131,072 processors
- **Download:**
  ```bash
  wget https://zenodo.org/records/8196385/files/BGL.zip?download=1 -O BGL.zip
  unzip BGL.zip -d BGL/
  ```
- **LOGHUB page:** https://github.com/logpai/loghub/tree/master/BGL
- **Original paper:** Adam J. Oliner, Jon Stearley, "What Supercomputers Say: A Study of Five System Logs" (DSN'07)

### Thunderbird

- **Lines:** 211,212,192 (29.60 GiB)
- **Time span:** 244 days
- **Labeled:** Yes (alert category tags — same format as BGL)
- **Source:** Thunderbird supercomputer at Sandia National Labs (SNL), 9,024 processors
- **Download:**
  ```bash
  wget https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1 -O Thunderbird.tar.gz
  tar xzf Thunderbird.tar.gz -C Thunderbird/
  ```
- **LOGHUB page:** https://github.com/logpai/loghub/tree/master/Thunderbird
- **Original paper:** Oliner & Stearley (DSN'07) — same as BGL

### Spirit

- **Lines:** ~272M (864 MB compressed)
- **Labeled:** Yes (alert category tags — same format as BGL/Thunderbird)
- **Source:** Spirit supercomputer at Sandia National Labs (SNL)
- **Download:**
  ```bash
  wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/spirit2.gz -O Spirit.gz
  gunzip Spirit.gz
  ```
- **Project page:** https://www.usenix.org/cfdr-data#hpc4
- **Original paper:** Oliner & Stearley (DSN'07) — same as BGL/Thunderbird
- **Note:** Spirit is part of the USENIX CFDR HPC4 collection. It is NOT in the Zenodo LOGHUB archive.

### Download with `datasets/download.py`

The repo ships a download script (pure stdlib, resumable, size-verified):

```bash
# All four corpora -> datasets/raw/
python datasets/download.py --extract

# Just the smaller corpora (HDFS 1.5 GiB + BGL 708 MiB)
python datasets/download.py --only hdfs bgl --extract

# Verify already-downloaded archives match the expected sizes
python datasets/download.py --verify

# Also fetch the 2k LOGHUB samples for smoke tests
python datasets/download.py --samples --only hdfs bgl thunderbird
```

Each corpus is downloaded to `datasets/raw/<ARCHIVE>` (e.g. `HDFS_v1.zip`)
and, with `--extract`, unpacked into `datasets/raw/<corpus>/`. Spirit is a
single-file gzip (`Spirit.gz` -> `datasets/raw/spirit/Spirit.log`). The
script resumes interrupted downloads via HTTP Range requests and keeps a
`.part` file on size mismatch so a retry continues where it left off.

### Manual download (alternative)

If you prefer to fetch by hand:

```bash
mkdir -p datasets/raw
wget https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1       -O datasets/raw/HDFS_v1.zip
wget https://zenodo.org/records/8196385/files/BGL.zip?download=1           -O datasets/raw/BGL.zip
wget https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1 -O datasets/raw/Thunderbird.tar.gz
wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/spirit2.gz -O datasets/raw/Spirit.gz

unzip    datasets/raw/HDFS_v1.zip      -d datasets/raw/hdfs/
unzip    datasets/raw/BGL.zip          -d datasets/raw/bgl/
tar xzf  datasets/raw/Thunderbird.tar.gz -C datasets/raw/thunderbird/
gunzip   datasets/raw/Spirit.gz        # -> datasets/raw/Spirit (rename to spirit/Spirit.log)
```

### Thunderbird / Spirit are large

Thunderbird is ~29.6 GiB and Spirit ~864 MiB compressed. For initial
development use `--only hdfs bgl` (≈2.2 GiB total); the pipeline is identical
across corpora. Both archives support resume, so a partial download can be
continued by re-running the same command.

---

## Small Samples (2k lines each)

The LOGHUB GitHub repo includes 2k-line samples for quick testing:

| Dataset | File |
|---|---|
| HDFS | `HDFS/HDFS_2k.log` |
| BGL | `BGL/BGL_2k.log` |
| Thunderbird | `Thunderbird/Thunderbird_2k.log` |

Clone the repo:
```bash
git clone https://github.com/logpai/loghub.git
```

These samples also include pre-parsed structured CSVs from Drain:
- `*_2k.log_structured.csv` — parsed log with EventId, EventTemplate, ParameterList
- `*_2k.log_templates.csv` — extracted event templates

---

## Preparing the Splits with `datasets/prepare.py`

After downloading + extracting the raw logs, build the temporal train/test
splits and the 12-field schema records:

```bash
# Default: 600k lines/corpus, last 200k = test, parsed with Drain
python datasets/prepare.py

# Subset + custom sample sizes
python datasets/prepare.py --only hdfs bgl --sample-size 600000 --test-size 200000

# Quick smoke test on the 2k samples (no Drain install required)
python datasets/prepare.py --smoke

# Use a regex fallback parser if logparser3 is not installed
python datasets/prepare.py          # auto-falls back; install logparser3 for full Drain
```

What `prepare.py` does, per corpus:

1. **Locate** the raw `.log` file under `datasets/raw/<corpus>/`.
2. **Reservoir-sample** `--sample-size` lines (default 600k) uniformly over
   the file. Memory is bounded to the sample size (works on 211M-line
   Thunderbird without loading it all).
3. **Temporal split**: the last `--test-size` lines of the sample (default
   200k) become the test set; the rest are train (matches the paper's
   "hold out most recent 200k per corpus" strategy).
4. **Parse with Drain** (via `logparser3`) for structural fields
   (`message_template`, `component`, `severity`, `host`,
   `timestamp_norm`, `error_code`, `affected_entity`, `metric_value`).
   If `logparser3` is not installed, falls back to a regex-based template
   detector so the pipeline still runs end-to-end.
5. **Native anomaly labels**: BGL/Thunderbird/Spirit carry alert tags in
   the first column ("- " = normal, anything else = anomaly); HDFS uses
   severity-based heuristics in the absence of block-level labels. These
   populate `is_anomaly` + `anomaly_score`.
6. **Emit** 12-field schema records as JSONL:

```
datasets/splits/
  train/<corpus>.jsonl      # 400k records/corpus
  test/<corpus>.jsonl       # 200k records/corpus
  manifest.json             # per-corpus counts + anomaly rates
datasets/parsed/<corpus>/   # Drain structured + templates CSVs
```

### Semantic field labels (placeholder)

The three semantic fields — `event_type`, `affected_entity`,
`remediation_hint` — are **not** recoverable from Drain or native labels.
The paper uses an unreleased teacher-distillation pipeline for these.
`prepare.py` emits placeholder values (`event_type` = first 30 chars of
the template, `remediation_hint` = `"none"`); replace them with
teacher-distilled labels when a teacher LLM is wired in (see issue #6).

### Manifest format

`datasets/splits/manifest.json` records what was produced:

```json
{
  "corpora": [
    {
      "corpus": "hdfs",
      "raw_path": "datasets/raw/hdfs/HDFS.log",
      "raw_lines": 11175629,
      "sample_size": 600000,
      "train_lines": 400000,
      "test_lines": 200000,
      "train_anomaly_pct": 2.87,
      "test_anomaly_pct": 2.93,
      "drain_available": true,
      "semantic_labels": "placeholder (teacher distillation not released)"
    },
    ...
  ],
  "schema": ["timestamp_norm", "host", "component", ...],
  "sample_size": 600000,
  "test_size": 200000
}
```

---

## Label Pipeline

The paper uses a two-stage label pipeline (part of the setup, not a benchmark itself):

### 1. Drain — Structural Field Labeling

- **What it does:** Online log parsing with fixed-depth tree — extracts event templates from unstructured logs
- **Repository:** https://github.com/logpai/logparser
- **Install:**
  ```bash
  pip install logparser3
  ```
- **Paper ref:** Pinjia He et al., "Drain: An Online Log Parsing Approach with Fixed Depth Tree" (ICWS'17)

**Usage:**
```python
from logparser.Drain import LogParser

input_dir = 'datasets/raw/HDFS/'
output_dir = 'datasets/parsed/HDFS/'
log_file = 'HDFS.log'
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
regex = [r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)']  # IP
st = 0.5    # similarity threshold
depth = 4   # depth of all leaf nodes

parser = LogParser(log_format, indir=input_dir, outdir=output_dir,
                   depth=depth, st=st, rex=regex)
parser.parse(log_file)
```

**Outputs:**
- `*_templates.csv` — EventId, EventTemplate, Occurrences
- `*_structured.csv` — parsed log with EventId, EventTemplate, ParameterList

### 2. Teacher Distillation — Semantic Field Labeling

- **What it does:** A teacher model labels a subset of logs for semantic fields (`event_type`, `affected_entity`, `remediation_hint`) that Drain cannot extract structurally
- **Method:** Standard knowledge distillation (Hinton et al., 2015)
- **Note:** Not released — part of the paper's methodology. Absolute F1 is bounded by teacher quality.

### 3. Native Alert Labels — Anomaly Ground Truth

- **BGL & Thunderbird:** Alert category tags in the first column of the log (from USENIX HPC4 collection)
- **HDFS:** Manually labeled through handcrafted rules (block-level anomaly labels)
- **Spirit:** Same alert tag format as BGL/Thunderbird

---

## Schema Definition

The paper reformulates all four corpora as a **12-field schema-filling task**:

| # | Field | Type | Category |
|---|---|---|---|
| 1 | `timestamp_norm` | datetime | Structural |
| 2 | `host` | string | Structural |
| 3 | `component` | string | Structural |
| 4 | `severity` | enum | Structural |
| 5 | `event_type` | string | Semantic |
| 6 | `message_template` | string | Structural |
| 7 | `error_code` | string | Structural |
| 8 | `affected_entity` | string | Semantic |
| 9 | `metric_value` | float | Structural |
| 10 | `is_anomaly` | bool | Anomaly |
| 11 | `anomaly_score` | float | Anomaly |
| 12 | `remediation_hint` | string | Semantic |

- Rendered as JSON: ~128 tokens
- Structural fields: labeled by Drain template parsing
- Semantic fields: labeled by teacher distillation
- Anomaly fields: from native alert labels (BGL/Thunderbird/Spirit) or handcrafted rules (HDFS)

---

## Expected Directory Structure

```
datasets/
  README.md               # This file
  download.py             # Fetch + extract raw corpora
  prepare.py              # Sample -> temporal split -> Drain -> 12-field JSONL
  raw/                    # Raw downloaded logs (gitignored)
    HDFS_v1.zip
    BGL.zip
    Thunderbird.tar.gz
    Spirit.gz
    hdfs/                 # Extracted
    bgl/
    thunderbird/
    spirit/
  parsed/                 # Drain-parsed output (gitignored)
    hdfs/
    bgl/
    thunderbird/
    spirit/
  splits/                 # Temporal train/test splits (gitignored)
    train/
      hdfs.jsonl
      bgl.jsonl
      thunderbird.jsonl
      spirit.jsonl
    test/
      hdfs.jsonl
      bgl.jsonl
      thunderbird.jsonl
      spirit.jsonl
    manifest.json
```

> **Note:** `raw/`, `parsed/`, and `splits/` are gitignored — only the
> scripts and this README are committed. Run `download.py` then
> `prepare.py` to materialize the data.

---

## What's NOT Available

The paper is a **preprint under review** — the following are NOT released:

- FlowFill model weights (1.7B masked diffusion LM)
- Preprocessing scripts (12-field schema reformulation, temporal split code)
- Teacher distillation pipeline (for semantic field labels)
- Evaluation scripts (Field-F1, Consistency, Schema-validity metrics)
- Serving infrastructure (Streaming/Packed mode implementation)

Only the **datasets** and the **Drain parser** are publicly available.

---

## References

1. **LOGHUB:** Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics." ISSRE, 2023. https://arxiv.org/abs/2008.06448
2. **Drain:** Pinjia He, Jieming Zhu, Zibin Zheng, Michael R. Lyu. "Drain: An Online Log Parsing Approach with Fixed Depth Tree." ICWS, 2017.
3. **HPC4 (BGL/Thunderbird/Spirit):** Adam J. Oliner, Jon Stearley. "What Supercomputers Say: A Study of Five System Logs." DSN, 2007. https://dblp.org/rec/conf/dsn/OlinerS07.html
4. **HDFS:** Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael Jordan. "Detecting Large-Scale System Problems by Mining Console Logs." SOSP, 2009.
5. **Knowledge Distillation:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531, 2015.
