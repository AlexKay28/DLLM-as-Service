# Validate performance with DiffusionGemma-26B via vLLM — size-matched AR comparison

**Type:** Performance validation experiment (deferred from GPU session 2026-08-11)
**Depends on:** #4 (latency), #5 (throughput), #3 (quality — LLaDA results as baseline)
**Status:** BLOCKED (updated 2026-08-12) — the CUDA-driver blocker below is
resolved (driver 580.65.06 / CUDA 13.0), but DiffusionGemma needs vLLM >= 0.24.0,
which pins torch 2.11 / CUDA 13, and this host's mirror has no CUDA-13
cudnn/nccl/cufile/cusparselt. The AR half of this issue is done under vLLM 0.19.1.
See `results/diffusiongemma_perf.md` and `results/vllm_serving.md`.

---

## 1. Motivation

During the 2026-08-11 GPU session, we validated quality claims (issue #3) using
LLaDA-8B-Instruct as the MDM substitute on HF transformers (naive serving, no
custom kernels). Results confirmed AR-vs-MDM near-parity (Field-F1 0.779 vs
0.779) but **absolute latency was ~55x slower than the paper's 23ms** (our
~1.4s/event at K=8) due to:

1. No KV-cache reuse across denoising steps (bidirectional attention — fundamental)
2. No FlashAttention-2 / PagedAttention serving infra
3. LLaDA-8B is 4.7x bigger than paper's FlowFill-MDM (1.7B)

vLLM v0.9.0+ added native MDM support, but **only for DiffusionGemma** (not
LLaDA). This issue tracks using DiffusionGemma via vLLM as a *second* MDM
substitute specifically for performance validation, paired with a size-matched
AR baseline for fair comparison.

## 2. The plan

### 2.1 Models (size-matched pair)

| Role | Model | Params | Active params | Serving |
|------|-------|--------|---------------|---------|
| MDM substitute | `google/diffusiongemma-26B-A4B-it` | 26B | 4B (MoE) | vLLM (native MDM runner) |
| AR size-match | `Qwen/Qwen2.5-3B-Instruct` (or `7B`) | 3B / 7B | — | vLLM (PagedAttention) |

**Why size-match on *active* params:** DiffusionGemma-26B-A4B uses MoE with
4B active parameters — the fair AR comparison is a ~3-4B dense AR model, not
a 26B dense one. If Qwen2.5-3B is used, both sides have ~3-4B active params.

**Alternative:** Use the FP8-quantized DiffusionGemma
(`RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`) to fit in 80GB and reduce
per-step cost.

### 2.2 What we measure

| Metric | Why |
|--------|-----|
| p50/p99/p99.9 latency (streaming m=1) | Table 3 — does vLLM-served MDM approach paper's 23ms? |
| Throughput @ 250ms p99 SLA (packed mode) | Table 4 — does MDM's batching advantage hold with proper serving? |
| `diffusion_steps` sweep (K=4,8,16,24) | Does the K=8 knee + linear latency hold on DiffusionGemma? |
| Quality (Field-F1, Schema-validity) | Sanity: can DiffusionGemma do our 12-field extraction task at all? |

### 2.3 vLLM serving setup

```python
from vllm import LLM, SamplingParams

llm = LLM(model="google/diffusiongemma-26B-A4B-it", trust_remote_code=True)
sampling_params = SamplingParams(
    extra_body={"diffusion_steps": 8, "entropy_threshold": 0.005}
)
outputs = llm.generate(prompts, sampling_params)
```

For AR:
```python
llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
sampling_params = SamplingParams(max_tokens=160, temperature=0)
```

## 3. Caveats to document

1. **Different denoising algorithm**: DiffusionGemma uses *block diffusion*
   (semi-AR blocks natively), not FlowFill's pure parallel masked diffusion.
   Comparing its latency to paper's 23ms is apples-to-oranges — different
   algorithm, different model architecture.
2. **Different model family**: DiffusionGemma is Gemma-based, not LLaDA-based.
   Quality on our 12-field log extraction task is unknown — it may not follow
   JSON instructions as well as LLaDA-Instruct. Need a quality sanity check
   before trusting performance numbers.
3. **MoE vs dense**: DiffusionGemma is MoE (4B active of 26B). The paper's
   FlowFill-MDM is dense 1.7B. Active-param matching is the fairest
   comparison, but FLOP characteristics differ.
4. **Two MDM substitutes in one report**: We'd report LLaDA-8B (quality focus,
   HF transformers) AND DiffusionGemma-26B (performance focus, vLLM) as
   complementary data points, clearly labeled.

## 4. Acceptance criteria

- [ ] DiffusionGemma loads in vLLM on the H100 (FP8 version if bf16 doesn't fit)
- [ ] Quality sanity check: DiffusionGemma produces valid JSON on >=10 BGL log
      lines (if not, document and close the issue — can't use it for perf if it
      can't do the task)
- [ ] Size-matched AR baseline (Qwen2.5-3B or 7B) loaded in vLLM
- [ ] Table 3 latency: p50/p99/p99.9 for both at matching active-param count
- [ ] Table 4 throughput: max batch @ 250ms p99 SLA for both
- [ ] K-sweep on DiffusionGemma (diffusion_steps = 4,8,16,24)
- [ ] Results saved to `results/diffusiongemma_perf.json` + `.md`
- [ ] Comparison report: our vLLM-served numbers vs paper's Table 3/4

## 5. Why this issue exists (not done in the 2026-08-11 session)

During the GPU session, we discovered:
- vLLM only supports DiffusionGemma for MDM (not LLaDA) — confirmed by
  inspecting `vllm/model_executor/models/registry.py` in the vLLM 0.26.0 wheel
- DiffusionGemma-26B is 3.2x bigger than LLaDA-8B — downloading + loading it
  would take significant time and GPU memory
- The primary quality results (issue #3) were already obtained with LLaDA-8B
- Switching MDM substitutes mid-session risked losing comparability

Deferred to a dedicated session with proper setup time.

## 6. Related

- #3 (quality — LLaDA-8B results as baseline for DiffusionGemma quality check)
- #4 (latency — this issue provides the vLLM-served data point)
- #5 (throughput — this issue provides the vLLM-served data point)
- #6 (ablations — DiffusionGemma's `diffusion_steps` param maps to K-sweep)
