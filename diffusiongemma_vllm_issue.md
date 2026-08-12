# Infer DiffusionGemma-26B fast via vLLM — unblock the CUDA-13 wheel gap

**Type:** Environment unblock + performance experiment
**Blocks:** #12 (the MDM half — the AR half is done, see `results/vllm_serving.md`)
**Status:** Blocked on package availability. Everything else is ready.

---

## 1. Goal

Serve `google/diffusiongemma-26B-A4B-it` under **vLLM** and get the Table 3 /
Table 4 numbers that no substitute has been able to produce so far. This is the
only path we have to a *properly served* MDM data point — LLaDA-8B has always
run on naive HF transformers, so every MDM-vs-AR performance comparison in this
repo is still confounded by serving. HF-transformers DiffusionGemma would not
fix that: it would give another serving-bound number and answer nothing.

The AR side of the comparison already exists, size-matched to DiffusionGemma's
~4B active params (`results/vllm_serving_3b.json`): **p50 662ms, peak 145 ev/s,
Field-F1 0.826**.

## 2. The blocker, precisely

`DiffusionGemmaForBlockDiffusion` first appears in **vLLM 0.24.0** (verified by
reading `models/registry.py` out of each wheel: absent in 0.19.1/0.21.0/0.22.1/
0.23.0, present in 0.24.0/0.25.1/0.26.0). Every version that has it pins
`torch==2.11.0`, a CUDA-13 build.

torch 2.11's `libtorch_cuda.so` hard-links four libraries the internal pip
mirror does not carry:

```
$ objdump -p torch/lib/libtorch_cuda.so | grep NEEDED
  NEEDED  libcudnn.so.9        <- missing
  NEEDED  libnccl.so.2         <- missing
  NEEDED  libcufile.so.0       <- missing
  NEEDED  libcusparseLt.so.0   <- missing
```

These are `DT_NEEDED`, resolved by the dynamic loader at import. No
dependency-override trick avoids them — torch 2.11 simply cannot import.

**vLLM 0.19.1 installs and works** on the cu12.8 stack (torch 2.10.0+cu128) and
serves Qwen2.5 on the H100 at full speed. It has no DiffusionGemma.

## 3. What this session established

- The old blocker is **gone**: driver is 580.65.06 / CUDA 13.0 (was 535 / 12.2).
- The mirror carries most of the CUDA-13 line under the **unsuffixed** names
  NVIDIA switched to (`nvidia-cublas` 13.6.1, `nvidia-cuda-runtime` 13.3.29,
  `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cufft`, `nvidia-curand`,
  `nvidia-cusolver`, `nvidia-cusparse`, `cuda-bindings`) — but **not**
  `nvidia-cudnn`, `nvidia-nccl`, `nvidia-cufile`, `nvidia-cusparselt`,
  `nvidia-nvjitlink`, `nvidia-nvtx`, `cuda-toolkit`, `cuda-pathfinder`.
- The mirror **proxies arbitrary versions of projects it knows** (fetched
  `nvidia-cublas==13.0.0.19` on demand) but 404s on unknown project names. So
  this is a project-registration gap, not a storage decision.
- The **NVIDIA apt repo is configured** on this host and its metadata lists
  exactly what we need — but the host cannot reach it
  (`connect: permission denied`, same egress block as pypi.org and
  download.pytorch.org):

  | package | version |
  |---|---|
  | `libcudnn9-cuda-13` | 9.25.0.15-1 |
  | `libnccl2` | 2.30.7-1+cuda13.3 |
  | `libcufile-13-0` | 1.15.1.6-1 |
  | `libcusparselt0` | 0.7.1.0-1 |

- vLLM needs one stub regardless of version: `openai_harmony` is imported
  unconditionally by `vllm/entrypoints/chat_utils.py`, which sits on the offline
  `LLM` import path though nothing there uses it. Working stub already written
  at `.venv-dllm/lib/python3.10/site-packages/openai_harmony.py`.
- Weights are **already cached** (49 GB bf16) — no download needed. Layout is a
  diffusers pipeline, not a plain transformers checkpoint:
  `DiffusionGemmaPipeline` + `Gemma4Processor` + `BlockRefinementScheduler`
  (`block_length` 32, `num_inference_steps` 32, `threshold` 0.95), with
  `canvas_length` 256, `max_denoising_steps` 48, `confidence_threshold` 0.005,
  `entropy_bound` 0.1. Expect the vLLM knobs to be `diffusion_steps` /
  `entropy_threshold` rather than a token budget.

## 4. Ways in, cheapest first

1. **Register the four wheels on the internal mirror.** Ask for
   `nvidia-cudnn`, `nvidia-nccl`, `nvidia-cufile`, `nvidia-cusparselt` (and, if
   torch's preload then complains, `nvidia-nvjitlink`, `nvidia-nvtx`,
   `cuda-pathfinder`). No code, no root, and the mirror already proxies versions
   on demand once a project is known. **Recommended.**
2. **Sideload from a machine with egress.** Copy in either the wheels or the
   four `.deb`s above. Neither needs root:
   `pip install ./nvidia_*.whl`, or `dpkg-deb -x pkg.deb $PREFIX` plus
   `LD_LIBRARY_PATH=$PREFIX/usr/lib/x86_64-linux-gnu`.
3. **Open egress** to `developer.download.nvidia.com` or
   `files.pythonhosted.org`.
4. **Backport DiffusionGemma into vLLM 0.19.1** (which already runs on cu12).
   Budget ~1h to first decide if it is even possible: unpack the 0.25.1 wheel
   and check whether the support is one model file registerable via
   `ModelRegistry.register_model()`, or whether it reaches into the engine
   (block-diffusion runner, non-causal masks inside a block, scheduler changes).
   If it touches the runner, abandon this route — do not half-port an engine.

**Not an option:** stubbing the four `.so` files. The symbols
`libtorch_cuda.so` imports would be undefined at load, and even if that were
papered over, a latency benchmark taken on faked CUDA libraries is not a
result.

## 5. Acceptance criteria

- [ ] vLLM ≥0.24.0 imports and initializes on the H100
- [ ] DiffusionGemma-26B loads (bf16, 49 GB, fits in 80 GB; the FP8 copy was
      reported corrupt on 2026-08-11 — re-verify before trusting it)
- [ ] Quality sanity check: valid 12-field JSON on ≥10 BGL lines. If it cannot
      do the task, record that and close — performance numbers from a model
      that cannot do the task are meaningless
- [ ] Table 3: p50/p99/p99.9 streaming (m=1), vs AR-3B's 662/1062/1336 ms
- [ ] Table 4: events/s across the SLA grid, vs AR-3B's 145 ev/s peak
- [ ] `diffusion_steps` sweep {4, 8, 16, 24, 32} — does the K=8 knee and the
      R²=1.0 latency-in-K linearity we saw on LLaDA hold under real serving?
- [ ] Results written to `results/diffusiongemma_perf.{json,md}`

`models/experiments/run_vllm_serving.py` already does all of this for AR and
needs only a DiffusionGemma code path — the data loading, prompt building,
metrics, SLA sweep and report generation are shared.

## 6. Watch out

- **Disk is at 98%** (4 GB free of 196 GB). A torch-2.11 venv needs ~10 GB. The
  reportedly-corrupt FP8 checkpoint is 26 GB and is the obvious thing to
  reclaim — but with no egress it cannot be re-downloaded, so verify before
  deleting.
- Keep the comparison **active-parameter matched**: DiffusionGemma-26B-A4B is
  MoE with ~4B active, so Qwen2.5-3B is the fair AR row, not the 7B.
- DiffusionGemma uses **block diffusion** (semi-AR blocks natively), not
  FlowFill's pure parallel masked diffusion, and it is Gemma-based rather than
  LLaDA-based. Any comparison to the paper's 23 ms is across a different
  algorithm and a different model family — label it as such.

## 7. Related

- `results/diffusiongemma_perf.md` — the blocker analysis in full
- `results/vllm_serving.md` — the AR half, plus the two AR bugs it uncovered
- `diffusiongemma_issue.md` — the original deferred issue
- `VALIDATION_REPORT.md` — carries the corrections this work forced
