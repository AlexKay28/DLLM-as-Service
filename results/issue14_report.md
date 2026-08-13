# Issue #14: rate-vs-lag crossover + output-length scaling — results

**Status:** RUN COMPLETE
**Date:** 2026-08-13 · **GPU:** A100 PCIe 80GB · **vLLM:** 0.25.1, torch 2.11.0+cu130
**Setup/data:** identical to results/diffusiongemma_perf.md (BGL test split, Poisson open-loop driver).

## Headline

Served dLLM (DiffusionGemma-26B, BF16) beats size-matched AR-3B **only inside
a bounded niche**: low-rate streams (λ ≲ 1.8 ev/s) and/or long outputs
(L ≥ ~65 tokens). Outside it, AR wins on every axis — and the DG engine has
a fatal concurrency bug in 0.25.1.

## P1: lag vs offered rate (open-loop Poisson, lag = arrival→finish)

**Corrects a #13 artifact first:** the "4.26 events/s throughput" reported
there came from lockstep `llm.generate` bursts. In real server mode
(`vllm serve`, Poisson arrivals) the DiffusionGemma engine **crashes** as
soon as requests mix phases (`num_running_reqs=2`):
`diffusion_gemma.py:96 — size of tensor a (257) must match size of tensor b (256)`
(see results/rate_lag_dg_engine_crash.json for the dead-engine sweep).
DiffusionGemma is therefore measured below with `--max-num-seqs 1` — its only
robust serving mode in 0.25.1; AR-3B runs the default 256-way scheduler.
This asymmetry is forced by the engine bug, not a choice.

| λ (ev/s) | DG p50 lag | DG p99 | AR-3B p50 lag | AR-3B p99 |
|---|---|---|---|---|
| 1 | 434 ms | 1167 ms | **725 ms** | 1293 ms |
| 1.5 | 566 ms | 2703 ms | — | — |
| 2 | 810 ms | 3613 ms | — | — |
| 2.5 | 4633 ms | 10574 ms | — | — |
| 4 | 21320 ms | 46612 ms (8 abandoned) | 719 ms | 1267 ms |
| 8 | — | — | 733 ms | 1284 ms |
| 32 | — | — | 893 ms | 1525 ms |
| 48 | — | — | 1231 ms | 1898 ms |
| 64 | — | — | 1783 ms | 2980 ms |
| 96 | — | — | 2828 ms | 4763 ms |
| 128 | — | — | 3932 ms | 6617 ms |

- **DG knee: λ ≈ 2.0–2.5 ev/s** — matches M/D/1 prediction for its ~420 ms
  service time (μ ≈ 2.4).
- **AR-3B knee: λ ≈ 64–96 ev/s** — batching absorbs load up to ~40x DG's.
- **p50 crossover ≈ 1.8 ev/s**: below it DG has lower lag (434 vs 725 ms at
  λ=1), above it AR and the gap opens catastrophically.
- Chart: `results/issue14_rate_lag.png`; data: `results/rate_lag_dg_seq1.json`,
  `results/rate_lag_ar3b.json`.

## P2: latency vs forced output length L (m=1, n=300)

| L (tokens) | DG p50 | AR-3B p50 | AR/DG |
|---|---|---|---|
| 32 | 416.9 ms | 210.6 ms | 0.51x (AR wins) |
| 64 | 418.5 ms | 411.5 ms | 0.98x |
| 128 | 418.1 ms | 812.6 ms | 1.94x |
| 256 | 419.4 ms | 1617.6 ms | 3.86x |
| 512 | 519.3 ms | 3226.4 ms | **6.21x** |

- AR: perfectly linear, **6.28 ms/token** (R²≈1.0, intercept ~0).
- DG: flat 418 ms across 32–256 (one canvas block — truly length-free
  within the canvas), +100 ms at 512 (second block). Stepwise as predicted.
- **Length crossover ≈ 65 tokens.** Our task's natural output is 132–171
  tokens — which is exactly why DG won the #13 m=1 comparison.
- Chart: `results/issue14_length_scaling.png`; data: `results/p2_*.json`.

## The defensible claim

On A100 + vLLM 0.25.1: **a served dLLM is strictly preferable when the
stream is slow (λ ≲ 1.8 ev/s) or outputs are long (≥ ~65 tokens, advantage
growing ~linearly to ≥6x at 512), and strictly worse otherwise** — plus a
robustness deficit (engine crash under mixed-phase concurrency; the 4.26
ev/s "capacity" from #13 is unreachable from a real API server). This
bounds the issue #10 niche quantitatively: news-to-report streams fit it
(low rate, long structured output); high-volume telemetry does not.

## Artifacts

| Path | Contents |
|---|---|
| `results/issue14_rate_lag.png` / `issue14_length_scaling.png` | charts |
| `results/rate_lag_dg_seq1.json` / `rate_lag_ar3b.json` | P1 sweeps |
| `results/rate_lag_dg_engine_crash.json` | the fatal-concurrency evidence |
| `results/p2_{dg,ar3b}_L{32..512}.json` | P2 cells |
| `models/experiments/rate_lag_curve.py` | open-loop driver (reusable) |
| `models/experiments/plot_issue14.py` | analysis/plots |

## Watch out

- Poisson arrivals understate real log burstiness; bursty rerun moves both
  knees left.
- Lag past capacity is transient-windowed (300-event windows, 45 s drain),
  not steady state — the shape (knee location) is the result, not the exact
  tail ms.
- DG quality remains F1 0.789 vs AR 0.822 (from #13) — the niche claim is
  performance-only.
- Upstream-reportable: the 257-vs-256 self-conditioning crash in vLLM
  DiffusionGemma under mixed-phase scheduling.
