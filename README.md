# DLLM-as-Service

## The Cost Shape of Block-Structured Decoding

Autoregressive (AR) and block-diffusion language models achieve comparable
quality on structured log extraction (9-field F1 within 0.002), but their
inference costs have fundamentally different shapes: AR cost is linear in
output length, while block-diffusion cost is a staircase with a
prefill-dominated floor. This repo contains the experiments, decision
procedure, and paper that establish these cost laws and determine where
each architecture belongs.

**Paper:** [`Cost-Shape-of-Block-Structured-Decoding.pdf`](Cost-Shape-of-Block-Structured-Decoding.pdf) (16 pages)

---

## Repository structure

```
.
├── Cost-Shape-of-Block-Structured-Decoding.pdf   # Compiled paper
├── CHECKLIST.md                                   # 5 harness defects (citiable)
├── CHANGELOG.md                                   # Project history
├── AGENTS.md                                      # Guide for AI agents
├── README.md                                      # This file
│
├── paper/                          # LaTeX source + figures
│   ├── main.tex                    # Entry point — compiles to 16 pages
│   ├── macros.tex                  # All numbers (shared with CHECKLIST.md)
│   ├── NUMBERS.md                  # Pinned stack, provenance, discrepancy analysis
│   ├── BACKBONE.md                 # Claim ledger (21 green, 2 amber, 0 red)
│   ├── CHARTS.md                   # Figure list with data sources
│   ├── verify_numbers.py           # Asserts all cited numbers against artifacts
│   ├── generate_figures.py         # Builds 7 figures from results/
│   ├── sections/                   # 9 .tex sections
│   └── figures/                    # 7 PDF/PNG figures
│
├── models/
│   ├── baselines/
│   │   ├── ar_model.py             # Prompt builder + JSON parser (first-object-wins)
│   │   ├── ar_wrapper.py           # AR model wrapper (HuggingFace)
│   │   ├── mdm_wrapper.py          # MDM/DiffusionGemma wrapper
│   │   ├── provenance.py            # Shared provenance helper (#57 R1)
│   │   └── gold_alignment.py        # Gold-alignment assertion + canary (#57 R6)
│   ├── experiments/
│   │   ├── run_vllm_serving.py     # Main AR/DG serving runner
│   │   ├── run_anchor_remeasure.py # 5×2 anchor re-measurement (#57 R2)
│   │   ├── run_input_length_sweep.py  # Input-length sweep (#44)
│   │   ├── run_perturbation.py     # Perturbation robustness (#27)
│   │   ├── run_energy_measurement.py # Energy at compute-matched pair
│   │   ├── run_memory_probe.py     # Long-context memory (#58 P5)
│   │   ├── run_infilling.py        # Infilling experiment (#59)
│   │   ├── analyze_*.py            # Analysis scripts
│   │   └── gpu_queue*.sh           # GPU experiment queues
│   └── decision/
│       ├── procedure.py            # Decision procedure (5 guards)
│       ├── cli.py                  # CLI: python -m models.decision
│       └── test_procedure.py       # 11/11 self-tests
│
├── metrics/
│   ├── schema.py                   # 12-field schema + 9-field subset
│   ├── quality_metrics.py          # Field-F1, schema-validity, consistency
│   └── latency_metrics.py          # Percentiles, throughput, energy
│
├── datasets/
│   ├── raw/                        # LOGHUB corpora (BGL, HDFS, Thunderbird, Spirit)
│   └── splits/test/                # Train/test splits
│
├── results/                        # All experiment artifacts (JSON + JSONL)
│   ├── anchors/                    # Anchor re-measurement + GPU queue logs
│   ├── llada_dinfer/               # LLaDA-8B under dInfer results
│   └── issue43/                    # Per-length predictions (#43)
│
├── docs/
│   ├── edr/                        # 22 Experimental Decision Records
│   │   ├── README.md               # Index + format guide
│   │   ├── 001-pin-the-stack.md
│   │   ├── ...
│   │   └── 022-adaptive-k-and-graceful-degradation.md
│   └── legacy/                     # Pre-correction reports (superseded)
│
└── logs/                           # Runtime logs
```

## Key results

| Metric | AR-3B | DG-26B-A4B |
|---|---|---|
| 9-field F1 (L=256) | 0.8516 | 0.8518 |
| Active parameters | 3.0B | 4.0B (top-8/128 MoE) |
| p50 at L=256 (batch-1) | 1616 ms | 378 ms |
| Batch multiplier | 77–86× | 1.18× |
| Input-length slope | 0.047 ms/tok | 0.123 ms/tok |

**DM wins:** per-event latency (2–10×), output-length invariance, energy (4.2 J vs 22.3 J)
**AR wins:** batched throughput (80× vs 1.18×), long input, deployment footprint (17× smaller)

## Decision procedure

```bash
python -m models.decision --L 512 --lambda 1.0 --no-batching --f1-tolerance 0.05
```

## Running experiments

```bash
# Pin the stack
export HF_HUB_OFFLINE=1
PY=.venv-dg/bin/python  # vLLM 0.25.1, torch 2.11.0+cu130, A100 80GB

# Anchor re-measurement
$PY models/experiments/run_anchor_remeasure.py --arm both --repeats 5 --n 300

# Input-length sweep
$PY models/experiments/run_input_length_sweep.py --n 100

# Verify all numbers
python3 paper/verify_numbers.py
```

## EDRs

Research decisions are recorded as [Experimental Decision Records](docs/edr/README.md)
— 22 records covering all 23 claims in the paper, with pre-registered predictions
and measured outcomes.
