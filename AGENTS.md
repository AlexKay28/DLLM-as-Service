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
| #3 quality | `run_quality_table2.py` ✅ | None — just run (AR or MDM via `--mdm`) |
| #4 latency | `run_latency_table3.py` ✅ | None — just run (AR or MDM via `--mdm`) |
| #5 throughput | `run_throughput_table4.py` ✅ | None — just run (AR or MDM via `--mdm`) |
| #6 ablations | `run_ablation_table5.py` ✅ | MDM denoising loop needs implementation in `mdm_wrapper.py` |

### MDM (FlowFill substitute) — needs GPU + weights

```bash
# MDM model (LLaDA-8B as FlowFill substitute)
python models/experiments/run_quality_table2.py \
  --mdm --model GSAI-ML/LLaDA-8B-Instruct --k-steps 8 --n 10000

# MDM latency (Table 3 FlowFill column)
python models/experiments/run_latency_table3.py \
  --mdm --model GSAI-ML/LLaDA-8B-Instruct --k-steps 8 --n 10000

# MDM throughput (Table 4 FlowFill column)
python models/experiments/run_throughput_table4.py \
  --mdm --model GSAI-ML/LLaDA-8B-Instruct --k-steps 8

# Ablations (Table 5 — needs MDM denoising loop implemented)
python models/experiments/run_ablation_table5.py \
  --model GSAI-ML/LLaDA-8B-Instruct --k-steps 8
```

**MDM wrapper:** `models/baselines/mdm_wrapper.py` — `MDMModel` class with:
- `extract()` — K-step parallel denoising
- `extract_with_blocks()` — Semi-AR B=1/2/4 (Table 5)
- `extract_with_k()` — K-sweep (K=8 knee)
- `MockMDMModel` for dry-runs (simulates paper's 23ms p50 @ K=8)

**BLOCKER:** `MDMModel.load()` and `_denoise()` raise `NotImplementedError`.
Need to implement LLaDA's denoising loop (see `mdm_wrapper.py` integration notes).
`MockMDMModel` works for pipeline testing.

### Energy measurement

`metrics/latency_metrics.py:energy_per_event` needs `avg_power_watts` (board-level
GPU power). Measure via:
```bash
nvidia-smi dmon -s p -d 1 -o D > power_log.csv  # run during inference
```
Then parse avg power and subtract idle baseline.
