# Issue: dLLM as the fast-answer engine — answer-latency under background batch load

---
Type: Experiment (research branch)
Extends: #14 (joint λ×L grid showed DG's win region is λ ≲ 2 and L ≥ 128)
Informs: #10, #11 — this is the quantitative core of the lag-critical niche
---

## 1. Goal / claim to prove

Issue #14 left DG with a bounded win region. This issue tests the sharpest
version of the dLLM value proposition:

> **When an answer is needed *at some moment* — immediately, regardless of
> what batch work is in flight — and overall RPS is secondary, a dLLM
> should deliver it with strictly lower lag than a size-matched AR**, because
> its service time is ~2x lower (420 vs 807 ms at our task lengths) and, with
> preemption, that gap is available *on demand*, mid-stream.

"Streaming" here means the paper's Table 3 notion (freshness of one
record's answer), not the throughput regime — in log terms: a FATAL line /
a news item arrives and its structured record must be fresh in <1 s while
the rest of the stream keeps flowing.

**Not the claim:** dLLM serves high-rate streams — #14 already refuted that
(capacity wall λ≈2 vs AR's ~79 ev/s, and no canvas batching planned upstream).

## 2. Design

Metric: **VIP lag** = arrival→finish of a rare latency-critical request
while a background Poisson stream runs. Chart: x = background λ, y = VIP
lag (p50 + p99.9), per model, with the SLA line (1 s).

Serving policies per model (each model gets its best policy):

| Model | Policy | Mechanism |
|---|---|---|
| DiffusionGemma | serialized + abort-on-VIP (shim, no engine changes) | on VIP arrival, client cancels in-flight requests (server aborts on disconnect), VIP runs at once, cancelled work resubmits after |
| AR-3B | vLLM `--scheduling-policy priority` (engine-native preemption, if honored in 0.25.1) | VIP submitted with high priority preempts running low-priority seqs (recompute later); if the flag is absent/no-op, fall back to the same abort shim |

Grids: DG background λ ∈ {0.5, 1, 1.5}; AR background λ ∈ {4, 8, 16, 32}
(each model probed across its own busy-but-stable region; DG cannot exceed
its λ≈2 capacity). VIP fires at fixed offsets (every ~30 s), n_vip ≥ 20 per
cell. Task = the standard BGL 12-field extraction; DG must not crash — the
abort shim doubles as a regression test of the mixed-phase workaround.

## 3. Prediction (falsifiable)

- DG VIP lag ≈ service time (≈420–850 ms incl. ≤1 aborted-service wait),
  flat in background λ.
- AR VIP lag ≈ residual wait + service ≈ 1.2–1.6 s at λ=8–16 (its busy
  region); priority preemption truncates the wait to ~current-step cost.
- If DG's VIP p99.9 stays under the SLA line while AR's crosses it → claim
  proven for the bounded niche: **"dLLM is the fast-answer lane"**.

## 4. Plan

| Step | Work |
|---|---|
| 0 | Probe vLLM 0.25.1: does `--scheduling-policy priority` exist and preempt? (read arg_utils + a quick AR run) |
| 1 | Extend `rate_lag_curve.py`: `--vip-every-s` + `--vip-abort` (shim policy) + record VIP vs background lag separately |
| 2 | Runs: DG {0.5,1,1.5} + AR {4,8,16,32} × VIP; ~15 min GPU total |
| 3 | Chart `vip_lag.png/.html`, results JSON, post to this issue, commit |

## 5. Acceptance criteria

- [ ] VIP lag curves for both models across background λ, p50 + p99.9
- [ ] Documented whether vLLM priority preemption is real in 0.25.1
- [ ] One-paragraph verdict: does DG deliver strictly-lower on-demand
      answer-latency under batch load, and at what background-load cost
      (aborted-and-redone work)?
- [ ] Chart + artifacts committed, posted here

## 6. Watch out

- Aborted background work must be counted (it is real cost) — report it.
- Priority preemption in vLLM recomputes aborted AR sequences — that shows
  up as background-latency inflation; record background p99 too.
- Same-host hardware labels (A100, vLLM 0.25.1) on every number.

## 7. Related

- #14 — the λ×L walls this builds on (DG serialized = its honest mode)
- Upstream: vllm-project/vllm#50699 (crash we hit + repro comment), #51994,
  #52015 (MRV2)
