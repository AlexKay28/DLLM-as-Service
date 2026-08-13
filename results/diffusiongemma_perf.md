# DiffusionGemma-26B via vLLM — runbook + results (issues #12/#13)

**Status:** RUN COMPLETE — all issue #13 acceptance criteria met.
**Date:** 2026-08-13

---

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA A100 80GB PCIe (sm_80) |
| Driver / CUDA | 580.65.06 / 13.0 |
| Python | CPython 3.10.12 (`/usr/bin/python3.10`) |
| venv | `.venv-dg` (uv 0.12.3) |
| vLLM | 0.25.1 |
| torch | 2.11.0+cu130 |
| transformers | 5.15.0 |
| triton | 3.6.0 |
| ninja | 1.13.0 (JIT build tool, required by vLLM at runtime) |
| Model | `google/diffusiongemma-26B-A4B-it` — 26B total, MoE ~4B active, BF16, 49 GB cached |
| Data | BGL test split: `datasets/splits/test/bgl.raw` + `bgl.jsonl` (same events as all earlier repo runs) |

Size-matched comparator: `Qwen/Qwen2.5-3B-Instruct` (≈4B active DG vs 3B AR).

## 2. Setup (from scratch)

```bash
uv venv .venv-dg --python 3.10
uv pip install --python .venv-dg/bin/python --index-url https://pypi.org/simple \
    vllm==0.25.1 openai-harmony ninja

# runtime env needed every session:
export PATH="$PWD/.venv-dg/bin:$PATH"   # engine subprocesses spawn `ninja`
export HF_HUB_OFFLINE=1                 # huggingface.co egress is blocked
```

Why not the internal mirror: `pypi.yandex-team.ru` carried `nvidia-cudnn` /
`nvidia-nccl` but still 404s `nvidia-cufile` / `nvidia-cusparselt` — both
hard-linked by torch 2.11's `libtorch_cuda.so`. Direct PyPI egress (newly
open as of 2026-08-13) resolves everything.

The `openai_harmony.py` stub that vLLM 0.19.1 needed is **not** required on
0.25.1 (imports fine without it).

## 3. Code changes

One file modified, `models/experiments/run_vllm_serving.py` (+17 lines):

- New flag `--denoising-steps K` → passes
  `diffusion_config={"canvas_length": 256, "max_denoising_steps": K}` to
  `LLM(...)`. Both keys are mandatory: a user-provided `DiffusionConfig`
  **replaces** checkpoint defaults wholesale (no merge), and `canvas_length`
  left None crashes the runner's buffer alloc
  (`TypeError: zeros(): ... got NoneType`). `canvas_length=256` comes from
  the checkpoint's `config.json`.
- The chosen K is recorded per-model in the output JSON.
- No other changes needed: prompt building (`build_chat_messages`), parsing
  (`parse_json_output`), metrics, and the SLA sweep are model-agnostic and
  worked on DiffusionGemma unchanged.

## 4. Failure log (order hit → fix)

| # | Failure | Root cause | Fix |
|---|---|---|---|
| 1 | FP8 checkpoint load: `humming.jit.compiler.NVRTCCompiler run failed` during weight repack | humming's FP8 repack kernel can't compile for sm_80 | Use BF16 checkpoint (49 GB). FP8 likely fine on sm_90 — untested |
| 2 | AR-3B engine init: `FileNotFoundError: 'ninja'` | vLLM 0.25 JIT-compiles kernels at startup via ninja, and the venv's bin dir wasn't on PATH for engine subprocesses | `pip install ninja` + `export PATH=...venv-dg/bin:$PATH` |
| 3 | Sweep run crash: `httpx.ConnectError: [Errno 13] Permission denied` (retries then dies in `ModelConfig`) | with `diffusion_config` set, vLLM enumerates repo files over the network; huggingface.co egress blocked | `HF_HUB_OFFLINE=1` (falls back to local snapshot path) |
| 4 | Sweep run engine crash: `zeros(): argument 'size' ... got NoneType` | user `diffusion_config` without `canvas_length` → None canvas | always pass `canvas_length: 256` alongside (see §3) |

## 5. Runs executed (in order)

```bash
S=models/experiments/run_vllm_serving.py; P=.venv-dg/bin/python

# 0. quality gate (issue §4): n=10 streaming only
$P $S --model google/diffusiongemma-26B-A4B-it --source bgl --n 10 \
      --skip-throughput --save-predictions ...  # PASS: F1=0.861, 100% valid

# 1. main Table 3 + Table 4
$P $S --model google/diffusiongemma-26B-A4B-it --source bgl --n 2000 \
      --output results/diffusiongemma_perf.json \
      --save-predictions results/diffusiongemma_preds.jsonl
      # logs/dg_full.log

# 2. same-host AR-3B comparator (old baseline was H100 — not comparable)
$P $S --model Qwen/Qwen2.5-3B-Instruct --source bgl --n 500 \
      --output results/diffusiongemma_ar3b_comparator.json
      # logs/ar3b_a100.log, logs/exp_batch2.log

# 3. denoising-steps sweep (streaming only)
for K in 8 32; do
  $P $S --model google/diffusiongemma-26B-A4B-it --source bgl --n 300 \
        --skip-throughput --denoising-steps $K \
        --output results/dg_steps_$K.json
done  # logs/exp_batch4.log
```

Sampling params everywhere: `temperature=0.0`, `max_tokens=256`,
dtype bfloat16, `max_model_len=4096`, `gpu_memory_utilization=0.85`,
prefix caching on (repo default). DG's own decoding config from
`generation_config.json`: confidence_threshold 0.005,
EntropyBoundSampler entropy_bound 0.1, max_denoising_steps 48 (default run).

## 6. Results

### Quality gate (n=10) — PASS

Field-F1 **0.861** · schema-valid 100% · consistency 100%.

### Table 3: streaming latency (m=1)

| Model | n | p50 (ms) | p99 (ms) | p99.9 (ms) | p99.9/p50 | Field-F1 | schema-valid |
|---|---|---|---|---|---|---|---|
| **DiffusionGemma-26B (BF16)** | 2000 | **420.2** | 710.4 | 1282.0 | 3.05x | 0.789 | 99.6% |
| AR-3B (same host) | 500 | 807.5 | 1299.6 | 1625.0 | 2.01x | 0.822 | 98.0% |
| AR-3B (old H100 run) | 2000 | 662.0 | 1062.1 | 1336.3 | 2.02x | 0.826 | 99% |
| HF transformers MDM LLaDA-8B (K=8) | — | 1389.0 | 1555.5 | 1648.2 | 1.19x | 0.779 | — |
| *paper: FlowFill-MDM (K=8)* | — | 23.0 | 31.0 | 36.0 | 1.57x | 0.907 | — |

Single stream: **DG ~1.9x faster p50 than the size-matched AR**, at −0.033
F1 and a heavier tail (3.05x vs 2.01x).

### Table 4: throughput burst sweep

DiffusionGemma:

| batch | wall (s) | p99 (ms) | events/s |
|---|---|---|---|
| 1 | 0.47 | 472 | 2.12 |
| 4 | 1.41 | 1408 | 2.84 |
| 16 | 3.76 | 3759 | 4.26 (peak) |
| 64 | 15.85 | 15849 | 4.04 |
| 256 | 61.49 | 61494 | 4.16 |

AR-3B (same host): 1.18 → 78.97 ev/s over B=1→256 (peak at B=256, p99 3242 ms).

**DG wall-clock is Linear-in-burst (~240 ms/request at any B): the vLLM
0.25.1 block-diffusion runner extracts no throughput from continuous
batching. DG loses ~18x to the size-matched AR on throughput.** Neither
model meets the paper's 250 ms p99 SLA at any batch (DG best: 484 ms @ m=1,
K=8).

### max_denoising_steps sweep (n=300, streaming)

| K | p50 (ms) | p99 (ms) | Field-F1 |
|---|---|---|---|
| 8 | 374.9 | 483.7 | 0.787 |
| 32 | 415.6 | 828.6 | 0.792 |
| 48 (default) | 420.2 | 710.4 | 0.789 |

The LLaDA law (latency ∝ K, R²=1.0, knee at K=8) **does not transfer**:
6x the step cap → +12% latency, ΔF1 ≤ 0.005. `max_denoising_steps` is a cap
and the EntropyBoundSampler (entropy_bound 0.1, conf 0.005) terminates
blocks far earlier. Operate at **K=8** purely for the better p99.

## 7. Caveats

- GPU changed since `vllm_serving.md` (H100 → A100): cross-file numbers are
  ~1.8x hardware-confounded (AR-3B: 662 ms/145 ev/s H100 vs 807 ms/79 ev/s
  A100). Only same-file DG-vs-AR-3B rows are directly comparable.
- All DG numbers are BF16 (FP8 blocked on sm_80, §4.1).
- DG ignores repetition/frequency/presence penalties (engine warning).
- DG emits a `thought` channel prefix and `<>` (not `<*>`) placeholders in
  message templates; `parse_json_output` handles both.
- DG is block diffusion (semi-AR), Gemma-based: comparison to FlowFill's
  23 ms is cross-algorithm, cross-family — label it as such.

## 8. Artifact inventory

| Path | Contents |
|---|---|
| `results/diffusiongemma_perf.json` | main run (n=2000) + merged sweep + comparator summary |
| `results/diffusiongemma_ar3b_comparator.json` | AR-3B same-host (n=500) |
| `results/dg_steps_{8,32}.json` | sweep runs (n=300 each) |
| `results/diffusiongemma_preds.jsonl` | per-event records + raw model text + latency |
| `logs/dg_full.log` · `logs/exp_batch2.log` · `logs/exp_batch4.log` | run logs |
