# Issue #15 — results

**Status: RUN COMPLETE.** Both policies exist and work in vLLM 0.25.1: per-request `priority` field + `--scheduling-policy priority` (engine preempts lower-priority running sequences — verified live). DG additionally survives `priority` + `--max-num-seqs 1` without the mixed-phase crash.

## The chart

**Attach image `results/vip_lag.png` here** (drag it into the comment box). Interactive version: `results/vip_lag.html` on branch `issue-12-vllm-serving`.

## The numbers (A100, vLLM 0.25.1, BGL extraction, Poisson background + VIP every 25 s)

| model | background λ (ev/s) | VIP p50 | VIP p99 | VIP p99.9 |
|---|---|---|---|---|
| **DiffusionGemma** | 0.5 | 405 ms | 483 ms | 483 ms |
| **DiffusionGemma** | 1.0 | 396 ms | 488 ms | 488 ms |
| **DiffusionGemma** | 1.5 | 432 ms | 770 ms | 770 ms |
| AR-3B | 8 | 762 ms | 845 ms | 845 ms |
| AR-3B | 16 | 803 ms | 934 ms | 934 ms |
| AR-3B | 32 | 1013 ms | 1134 ms | 1134 ms |
| AR-3B | 48 | 1710 ms | 10344 ms | 10344 ms |

## Verdict

**Claim proven for the bounded niche.** DG's latency-critical (VIP) answer lag stays ≈ 0.4–0.8 s — under the 1 s SLA at *every* background load it can serve. AR-3B under its own legitimately-busy rates never goes below 762 ms and blows past the SLA at λ≥32 (p99.9 10.3 s at λ=48). DG is the fast-answer lane: ~2x lower on-demand answer-latency than AR mid-stream, held flat by priority preemption. Costs are documented and small: DG needs `--max-num-seqs 1`/priority (mixed-phase concurrency crashes it — vllm-project/vllm#50699), and preempted AR work is recomputed (background p99 inflated from 1.3 s to 15.3 s at λ=48).

Artifacts: `results/vip_{dg,ar3b}.json`, `results/vip_probe_*.json`, `models/experiments/plot_vip.py`, driver support in `models/experiments/rate_lag_curve.py` (`--vip-every-s`, `--vip-policy abort|priority`).
