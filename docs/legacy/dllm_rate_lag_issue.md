# Issue: When does a served dLLM actually beat AR? Rate-vs-lag crossover + output-length scaling

---
Type: Experiment (follow-up to #13)
Blocks: #10, #11 (both need the quantitative claim this produces)
Depends on: #13 (done — DiffusionGemma serving works, numbers in results/diffusiongemma_perf.md)
---

## 1. Goal

Issue #13 established a narrow result: DiffusionGemma-26B (BF16, vLLM 0.25.1,
A100) beats size-matched AR-3B on single-stream latency (p50 420 vs 807 ms,
1.9x) but loses throughput ~18x (4.26 vs 79.0 events/s) and slightly on
quality (F1 0.789 vs 0.822). That is **not enough** to claim "dLLM is better
for log streaming" — and two specific experiments would make the claim
either defensible or dead:

- **P1 — rate-vs-lag curve.** X: offered stream rate (events/s). Y: lag
  (end-to-end latency, arrival→finish). Hypothesis to test: dLLM sits below
  AR at low load. Expect a **crossover**: DG's lower service time wins until
  its ~4 ev/s capacity, where its lag explodes while AR holds to ~79.
- **P2 — output-length scaling.** AR latency is linear in generated tokens
  (one forward pass per token). DG cost is stepwise — one canvas block
  covers up to 256 tokens. Hypothesis: DG's advantage grows with output
  length up to canvas size (256), then holds at a constant factor; at 256+
  token generations DG should win decisively.

## 2. Why the #13 evidence is insufficient

1. Our "throughput" numbers are closed-burst wall-clock, which **cannot
   measure queueing lag** (coordinated omission): every burst arrives at
   t=0, so there is no arrival-vs-finish distinction. A rate-vs-lag curve
   needs open-loop arrivals at controlled rates.
2. Different n per model (2000 DG vs 500 AR-3B), no confidence intervals,
   no paired per-event comparison.
3. Single dataset (BGL), one model pair, one GPU variant — the claim needs
   at least the four log sources the runner already supports.

## 3. Experiment P1: open-loop rate-vs-lag

### 3.1 Tooling (new code)

`models/experiments/rate_lag_curve.py`:

- Serve the model with `vllm serve` (OpenAI-compatible) in a subprocess.
- Drive it with an **open-loop** asyncio client: Poisson arrivals
  (exponential inter-arrival) at a fixed rate λ; per request record
  t_send (client), t_first_token, t_finish; **lag = t_finish − t_send**
  (arrival-referenced, avoids coordinated omission).
- Each λ runs for a fixed window (default 120 s or ~300 completed events,
  whichever first). Requests still in flight at window end are recorded as
  abandoned — that backlog **is** the measurement past capacity.
- Rate grid per model, bracketing the known knees:
  DG: {0.5, 1, 2, 3, 4, 5, 6, 8, 12} ev/s; AR-3B: {1, 4, 8, 16, 24, 32,
  48, 64, 96, 128} ev/s.
- Same BGL events for both models (paired windows: identical arrival-time
  seeds and event sequences per λ).
- Output: p50/p99 lag per λ per model + % abandoned → curve + JSON.

Prediction (falsifiable): DG lag < AR lag for λ ≲ 3–4 ev/s; DG curve
explodes at λ ≈ 4–5; AR flat to λ ≈ 60–70. **Crossover ≈ 4–6 ev/s** — i.e.
DG wins only low-rate streams. If DG stays below AR across the whole range,
vLLM's diffusion batching has improved and the throughput conclusion from
#13 gets rewritten.

### 3.2 Runtime estimate

DG: 9 rates × 120 s ≈ 18 min + load. AR-3B: 10 rates × 120 s ≈ 20 min +
load. Total ≈ 45 min GPU.

## 4. Experiment P2: output-length scaling

### 4.1 Design

- Add `--min-tokens` to `run_vllm_serving.py` (one-line SamplingParams
  change) so output length L is **forced**, independent of when EOS would
  natively occur: `max_tokens = min_tokens = L`.
- Sweep L ∈ {64, 128, 256, 512} × {DG, AR-3B}, m=1 streaming, n=300.
- Note: `min_tokens` changes task semantics (pads past the JSON), so this
  measures **decode mechanics**, not quality — report quality only for the
  natural-length runs already in #13.
- Per L, report p50/p99 latency; fit AR slope (ms/token) and locate DG's
  canvas-block steps (expect jumps at 257, 513).

Prediction (falsifiable): AR ≈ a + b·L with b ≈ 16 ms/token (807 ms ÷ ~50
tokens at m=1) → at L=256 ≈ 4.3 s, at L=512 ≈ 8.2 s. DG ≈ 420 ms per
canvas block + early-exit savings → at L=256 ≈ 0.45–0.9 s, at L=512 ≈ 0.9 s
(two blocks). So DG ≈ 5–10x faster at L=512. The crossover vs AR should sit
near L ≈ 25–30 tokens if DG's per-block cost is flat — **measure, don't
trust this arithmetic.** If DG's cost is NOT stepwise-in-canvas (e.g. it
scales in L too), the length-scaling argument for dLLMs is much weaker than
the literature implies.

### 4.2 Runtime estimate

4 lengths × 300 events ≈ 300 × (0.4–8 s) ≈ 25 min per model; ≈ 55 min GPU
total.

## 5. Statistics (both experiments)

- Identical event sequences for both models (paired design).
- n=1000 per condition where the run allows (P1 windows may be shorter).
- Bootstrap 95% CI on the p50-latency ratio DG/AR per condition; report
  per-event paired deltas, not just marginals.
- Sanity: warm up 50 requests before recording (first-request JIT spikes
  polluted the #13 p99.9 at small n).

## 6. Resolution plan

| Step | Work | Timebox |
|---|---|---|
| 0 | `--min-tokens` flag in run_vllm_serving.py; smoke test both models at L=64, n=10 | 30 min |
| 1 | Write rate_lag_curve.py (server subprocess + asyncio open-loop driver + JSON emit); smoke test on AR-3B at λ=1, 60 s | 1 h |
| 2 | P2 full sweep (§4) | 55 min GPU |
| 3 | P1 full sweep (§3) | 45 min GPU |
| 4 | Analysis: curves, crossover points, slope fits, CIs; charts (matplotlib in .venv-dg; md tables as fallback) | 1 h |
| 5 | Write results/rate_lag_curve.{json,md} + results/length_scaling.{json,md}; post to #10, #11, #12 | 30 min |

Steps 2 and 3 are interchangeable in order; run P2 first (reuses proven
tooling, de-risks the new open-loop code with a known-good GPU session).
Plotting: install `matplotlib` into `.venv-dg` only (experiment venv, not
the stdlib repo).

## 7. Acceptance criteria

- [ ] P1: lag-vs-rate curves for DG and AR-3B, same host, paired arrivals;
      crossover λ identified or "no crossover below capacity" stated
- [ ] P1: coordinated-omission-free lag (arrival-referenced) + abandoned-%
      reported past capacity
- [ ] P2: latency vs L ∈ {64,128,256,512} for both models; AR per-token
      slope fitted; DG block-step structure confirmed or refuted
- [ ] Bootstrap CIs on the p50 ratio at every condition; matched event sets
- [ ] Charts + JSON artifacts in results/
- [ ] One-paragraph defensible claim, bounded to the niche it supports
      (feeds #10's lag-critical framing and #11's survey)

## 8. Watch out

- DG's capacity is ~4.26 ev/s: past that, its queue grows unboundedly
  within a window. Keep windows fixed-length and record abandoned % rather
  than letting vLLM's queue explode.
- Label every number with host (A100) + vLLM 0.25.1 — vllm_serving.md's AR
  table is H100 and must not be mixed in.
- DG ignores penalties; its `thought` channel prefix and `<>` placeholders
  are handled by parse_json_output already.
- Poisson arrivals understate real log-stream burstiness; note it as a
  limitation (a bursty/Pareto rerun would move both knees left).
- `diffusion_config` replaces checkpoint defaults — keep passing
  `canvas_length: 256` (see #13 §6).

## 9. Related

- #13 — serving infrastructure + the narrow m=1 result this extends
- #10 — lag-time-critical workloads: P1's crossover directly bounds that niche
- #11 — use-case survey: P1/P2 supply its quantitative core
- results/diffusiongemma_perf.md — setup recipe + failure log (FP8/sm_80,
  ninja, HF_HUB_OFFLINE, canvas_length)
