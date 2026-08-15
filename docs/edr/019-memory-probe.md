# EDR-019: Long-context memory probe (inconclusive)

**Status:** Inconclusive
**Date:** 2026-08-15
**Issues:** #58 P5, #63

## Hypothesis

At long context and batch 1, the dLLM's absence of a growing KV cache may
give it a flatter memory curve than AR.

## Design

Measure `torch.cuda.max_memory_allocated()` for AR-1.5B at context lengths
1k/4k/16k/32k with `enforce_eager=True` and `gpu_memory_utilization=0.3`.

## Prediction

AR-1.5B memory grows linearly with context length (KV cache scales with
sequence length). DG memory is flat (no KV cache, fixed canvas).

## Result

Memory measurement returned 0.0 GB for all context lengths — vLLM manages
its own memory pool internally and `torch.cuda.max_memory_allocated()` does
not see it. The latency data IS useful:

| Context (tokens) | Latency (ms) |
|---|---|
| 1024 | 43.8 |
| 4096 | 83.9 |
| 16384 | 435.9 |
| 32768 | FAILED (model max_position_embeddings=32768) |

AR-1.5B latency grows ~10× from 1k to 16k context — consistent with the
prefill cost growing linearly with input length.

**Artifacts:** `results/issue58_memory_probe.json`

## Decision

**Inconclusive:** the memory measurement method was wrong (vLLM internal
pool vs torch allocator). The latency data confirms AR's input-length
sensitivity but does not answer the memory question.

A correct measurement would use `nvidia-smi --query-gpu=memory.used` during
inference, or set `gpu_memory_utilization` low enough that the pool is not
the binding constraint and measure torch allocations directly.

## Impact

- §9 threat: "long-context memory not measured; AR-1.5B latency grows 10×
  from 1k to 16k context, suggesting KV cache growth, but memory was not
  directly observed."
- This does not change the paper's conclusions — the deployment comparison
  (EDR-007) already shows AR-1.5B is 17× smaller at the quality-matched pair.
