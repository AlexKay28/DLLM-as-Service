# AGENTS.md

## Commands

No build step. Pure Python stdlib project (no pyproject.toml).

- **Lint/typecheck:** none configured.
- **Verify scripts:** `python3 -c "import py_compile; py_compile.compile('<file>', doraise=True)"`

## GPU Experiments (issues #3, #4, #5, #6)

These require a GPU and HuggingFace models. Prep code is committed; only model
loading + inference needs the GPU.

### Quick start (tomorrow)

```bash
# Install deps
pip install transformers accelerate torch

# Issue #3: Quality (Table 2) — run Qwen2.5-1.5B on BGL
python models/experiments/run_quality_table2.py \
  --model Qwen/Qwen2.5-1.5B-Instruct --n 10000

# With 7B too
python models/experiments/run_quality_table2.py \
  --model Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct --n 10000

# Dry-run (no GPU, verifies pipeline)
python models/experiments/run_quality_table2.py --mock --n 500
```

### Infrastructure built (CPU, already committed)

- `models/baselines/ar_model.py` — prompt builder + JSON output parser
- `models/baselines/ar_wrapper.py` — AR model wrapper (loads HF model, times
  each event, parses JSON → Record). Includes `MockARModel` for GPU-less testing.
- `models/experiments/run_quality_table2.py` — Table 2 experiment runner.
  Computes Field-F1, Schema-validity, Consistency, Anomaly-F1.
  Writes `results/quality_table2.json` + `.md`.

### What each GPU issue needs

| Issue | Script | Additional prep needed |
|-------|--------|----------------------|
| #3 quality | `run_quality_table2.py` ✅ | None — just run |
| #4 latency | (not yet built) | Build `run_latency_table3.py` using `ar_wrapper.py` — m=1 streaming, 10k events, p50/p99/p99.9 + J/event |
| #5 throughput | (not yet built) | Build `run_throughput_table4.py` — batch sweep [1..128], find max-batch @ 250ms p99 |
| #6 ablations | (not yet built) | K-sweep for MDM; needs FlowFill/LLaDA weights — most blocked |

### Energy measurement

`metrics/latency_metrics.py:energy_per_event` needs `avg_power_watts` (board-level
GPU power). Measure via:
```bash
nvidia-smi dmon -s p -d 1 -o D > power_log.csv  # run during inference
```
Then parse avg power and subtract idle baseline.
