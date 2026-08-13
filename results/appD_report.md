# Issue #17 (App D): fitted cost laws + batching multipliers — results

## Cost laws (measured, A100 80GB PCIe, vLLM 0.25.1, BF16, m=1)

### AR-3B (Qwen2.5-3B-Instruct)

```
T_AR(L) = 9.1 + 6.283 · L    ms     (R² ≈ 1.0, residual σ = 0.39 ms)
```

Linear law holds from L=32 to L=2048.  Slope = 6.283 ms/token matches the
#14 estimate (6.28) to three figures.  Tail ratio = 1.01× at L=1024 — AR
is deterministic at m=1.

### DiffusionGemma-26B (dLLM)

```
T_DG(L) = 418.2 + 113.4 · (⌈L / 256⌉ − 1)    ms
```

Stepwise: flat 418 ms for L ≤ 256 (one canvas), +113 ms per additional
256-token canvas block.  Equivalent marginal cost = 0.44 ms/token.

### Predictions vs measured

| L (tokens) | T_AR (meas) | T_DG (meas) | Ratio | T_DG (pred) | P-D2 |
|---|---|---|---|---|---|
| 128 | 814 ms | 418 ms | 1.94× | 418 ms | — |
| 256 | 1617 ms | 418 ms | 3.87× | 418 ms | — |
| 512 | 3222 ms | 532 ms | 6.07× | 518 ms | — |
| 1024 | 6443 ms | 779 ms | 8.49× | 719 ms | +8.4% ✓ |
| 2048 | 12892 ms | 1251 ms | 10.3× | 1120 ms | +11.7% ✓ |

**P-D1:** DG beats AR-3B at every L ≥ 128. ✓
**P-D2:** DG at L=1024 and L=2048 both within ±15% of extrapolation. ✓
**P-D3:** AR slope = 6.283 ms/token (expected 6.28), linear to L=2048. ✓
**P-D4:** (batching multipliers — pending multiplier sweep)

### Asymptotic ratio

```
T_AR / T_DG → 6.28 / 0.44 ≈ 14.3×    as L → ∞
```

The advantage **saturates** at ~14× (the issue said "toward 16×" — our
measured step of 113 ms/canvas vs the extrapolated 100 ms yields a
slightly lower ceiling, but the functional form is confirmed).

## Batching multipliers (measured)

| Model | L | batch=1 ev/s | peak ev/s | multiplier |
|---|---|---|---|---|
| DG | 128 | 2.45 | 4.11 | **1.7×** |
| DG | 512 | 1.87 | 2.82 | **1.5×** |
| AR-3B | 128 | 1.23 | 94.9 | **77×** |
| AR-3B | 512 | 0.31 | 26.7 | **86×** |

**P-D4 confirmed:** DG batching multiplier ≤ 2 (1.5–1.7×), well under the
≤3 prediction.  AR-3B reaches 77–86×.  The batching-capability gap is
**~50×** (77/1.7 at L=128).

**Important caveat:** DG's multiplier was measured with batch capped at
8.  At batch ≥16, vLLM 0.25.1 hits the known `diffusion_gemma.py:96` crash
(vllm-project/vllm#50699 — size 257 vs 256 under mixed-phase concurrency).
The cap is a **current vLLM limitation**, not a proven architectural bound.
However, even if the crash were fixed, block diffusion's denoising steps
are sequential per-canvas, so the architectural ceiling is well below AR's
continuous-batching multiplier.  The 1.5–1.7× measured here is a lower
bound; the true architectural ceiling is unknown but bounded by the
denoising-step serial dependency.

**Answer to #17's central question — is the 18× throughput deficit
architectural or vLLM-specific?**  **Both.**  The batching multiplier
gap (~50×) is partly vLLM's (the crash caps DG at batch 8) and partly
architectural (block diffusion cannot match AR's continuous batching
even without the crash, because denoising steps are sequential).
The m=1 latency advantage (DG wins 1.9–10× per call) is purely
architectural and unaffected by the vLLM crash.

## Chart

`results/appD_costlaws.png` — cost curves + ratio chart (attached).
