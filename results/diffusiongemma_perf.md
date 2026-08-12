# Issue #12: DiffusionGemma-26B via vLLM — blocker analysis

**Status:** NOT RUN — blocked on CUDA-13 wheel availability, not on hardware
**Date:** 2026-08-12 (supersedes the 2026-08-11 version of this file)

The AR half of issue #12 did complete, under real vLLM serving. Those results
are in [`vllm_serving.md`](vllm_serving.md). This file records only why the
DiffusionGemma half is still not runnable here.

---

## 1. The old blocker is gone

The 2026-08-11 version of this file reported:

> DiffusionGemma-26B requires CUDA 12.3+ driver. Our H100 system has CUDA 12.2.
> vLLM 0.9+ fails with "NVIDIA driver too old (found version 12020)".

That is fixed. The host now runs **driver 580.65.06 / CUDA 13.0** (was
535.129.03 / CUDA 12.2), and vLLM initializes on the H100 without complaint —
verified by serving Qwen2.5-1.5B/3B/7B through it for the results in
`vllm_serving.md`. The recommended next step from that report ("upgrade CUDA
driver to 12.3+ — sysadmin task") is done.

## 2. The blocker that replaced it

DiffusionGemma is reachable only through a vLLM version this host cannot
install.

**Which vLLM versions have it.** Read from `models/registry.py` inside each
wheel:

| vLLM | `DiffusionGemmaForBlockDiffusion` | torch pin |
|---|---|---|
| 0.19.1 | no | 2.10.0 (cu12.8) |
| 0.21.0 | no | 2.10.x |
| 0.22.1 | no | 2.11.0 |
| 0.23.0 | no | 2.11.0 |
| **0.24.0** | **yes** | **2.11.0** |
| 0.25.1 | yes | 2.11.0 |
| 0.26.0 | yes | 2.11.0 |

So DiffusionGemma needs vLLM >= 0.24.0, and every such version pins
`torch==2.11.0`, which is a CUDA-13 build.

**Why torch 2.11 will not install.** The internal mirror
(`pypi.yandex-team.ru`) is the only reachable index — `pypi.org` and
`download.pytorch.org` both fail to connect. It carries the complete CUDA-12
wheel line but only part of the CUDA-13 line. NVIDIA dropped the `-cu13`
suffix in the CUDA-13 packaging, and the unsuffixed names are split:

| Present | Absent |
|---|---|
| `nvidia-cublas`, `nvidia-cuda-runtime`, `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cufft`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`, `cuda-bindings` | `nvidia-cudnn`, `nvidia-nccl`, `nvidia-cufile`, `nvidia-cusparselt`, `nvidia-nvjitlink`, `nvidia-nvtx`, `cuda-toolkit`, `cuda-pathfinder` |

The absent ones are not optional extras that a flag can skip —
`libtorch_cuda.so` lists four of them as `DT_NEEDED`:

```
$ objdump -p torch/lib/libtorch_cuda.so | grep NEEDED
  NEEDED  libcudnn.so.9
  NEEDED  libnccl.so.2
  NEEDED  libcufile.so.0
  NEEDED  libcusparseLt.so.0
```

The dynamic loader resolves those at load time, so torch 2.11 cannot import at
all, and no amount of dependency-override surgery changes that. Installing
vLLM 0.25.1 with the missing deps overridden away confirms it:

```
OSError: libcudart.so.13: cannot open shared object file
ValueError: libcublasLt.so.*[0-9] not found in the system path
```

**What does work.** vLLM **0.19.1** resolves onto the full cu12.8 stack
(torch 2.10.0+cu128) with only genuinely optional kernel extras stripped —
`flashinfer-python`, `flashinfer-cubin`, `nvidia-cutlass-dsl`, `quack-kernels`,
`nvidia-cudnn-frontend`, `cuda-pathfinder`, `pynvvideocodec`,
`model-hosting-container-standards`, `openai-harmony`. It serves models
correctly on the H100. It has no DiffusionGemma.

One stub was needed: `openai_harmony` is imported unconditionally by
`vllm/entrypoints/chat_utils.py`, which sits on the import path of the offline
`LLM` API, though nothing on that path uses it (every real use is the
OpenAI-compatible server, the MCP bridge, and the gpt-oss response parser). The
stub is at `.venv-dllm/lib/python3.10/site-packages/openai_harmony.py` and
raises if anything ever calls into it.

## 3. What it would take

Any one of these unblocks it:

1. **Mirror the missing CUDA-13 wheels** — `nvidia-cudnn`, `nvidia-nccl`,
   `nvidia-cufile`, `nvidia-cusparselt` (plus `nvidia-nvjitlink`,
   `nvidia-nvtx`, `cuda-toolkit`, `cuda-pathfinder`). This is the clean fix and
   is an ask to whoever runs the mirror, not a code change.
2. **Outbound HTTPS to pypi.org or download.pytorch.org** from this host.
3. **A CUDA-13 torch built against system libraries** rather than wheel-shipped
   ones — i.e. build torch 2.11 from source against a local CUDA 13 toolkit.
   The host has CUDA 12.1/12.2 toolkits only, so this needs a toolkit install
   first.

Not viable: stubbing the four `DT_NEEDED` libraries. The symbols
`libtorch_cuda.so` imports from them would be undefined at load, and even if
that were papered over, benchmarking on faked CUDA libraries would not produce
numbers worth reporting.

## 4. Model weights are ready

Both checkpoints are already in the local HF cache, so nothing needs
downloading when the environment is fixed:

- `google/diffusiongemma-26B-A4B-it` — 49 GB, BF16
- `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` — 26 GB (reported corrupted
  in the 2026-08-11 session; re-verify before use)

BF16 fits in the H100's 80 GB.

## 5. Withdrawn results

The previous `results/diffusiongemma_perf.json` has been deleted and
`models/experiments/run_diffusiongemma_perf.py` is disabled. That script
reported p50 = 0.42 ms and 322k events/s for an 8B model — impossible figures,
caused by a silent fallback to synthetic log lines when it looked for BGL data
at a path this repo does not use, and by a "throughput" measurement that was a
sequential `for` loop rather than a batch. Details are in the script's
docstring.

The LLaDA-8B latency and K-sweep numbers quoted in the old version of this file
were *not* from that script — they came from issue #3 and #6 runs on real BGL
data (`results/quality_table2_mdm_llada_n2000.json`,
`logs/ablation_ksweep_n30.log`) and remain valid. But see `vllm_serving.md`:
the conclusion that MDM is 1.33x faster than AR was drawn against an
HF-`generate()` AR baseline and does not survive serving AR properly.

---

*Related: [`vllm_serving.md`](vllm_serving.md) (AR results), `diffusiongemma_issue.md` (original issue), `VALIDATION_REPORT.md`.*
