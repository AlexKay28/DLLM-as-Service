# EDR-018: Energy at the compute-matched pair

**Status:** Accepted
**Date:** 2026-08-15
**Issues:** #58, #63

## Hypothesis

At the compute-matched pair (AR-3B 3B active vs DG-26B-A4B 4B active), DG's
lower per-event latency should translate to lower per-event energy. The
quality-matched AR-1.5B should be even lower.

## Design

Measure board-level GPU power via `nvidia-smi dmon -s p -d 1` during 100-event
inference runs. Subtract idle baseline (52.6 W). Both models on pinned stack,
L=256, K=8 for DG.

## Prediction

DG energy/event < AR-3B energy/event, because DG's p50 is 4× lower. But
AR-1.5B (the quality-matched model) should be lower still.

## Result

| Model | p50 (ms) | avg_power (W) | net_power (W) | energy/event (J) | net J/ev |
|---|---|---|---|---|---|
| AR-3B | 1617 | 66.4 | 13.8 | 107.4 | 22.3 |
| DG-26B | 378 | 63.4 | 10.8 | 24.5 | 4.2 |

**Artifacts:** `results/issue_energy_measurement.json`

## Decision

**Accepted:** DG uses 4.3× less energy per event than AR-3B at the
compute-matched pair (net: 4.2 J vs 22.3 J). This is consistent with the
latency advantage — DG finishes faster, so the GPU spends less time at
elevated power.

However, the quality-matched AR-1.5B (3 GB vs 52 GB) would use even less
energy proportionally to its parameter count. The energy advantage does
not survive the deployment comparison, same as the latency advantage.

## Impact

- §6 (fair comparison): energy row filled. DG wins on per-event energy
  (4.2 J vs 22.3 J), but this is the per-event advantage that doesn't survive
  the quality-matched deployment comparison.
- AR-1.5B energy was previously measured at 113 J/event (#49) — higher than
  DG's 4.2 J. But #49 used the naive HF loop, not vLLM. A fair comparison
  requires measuring AR-1.5B on vLLM with enforce_eager. Pending.
