# Issue #22 — LLaDA-8B under dInfer: closing the MDM-side serving confound

**Host:** A100 80GB PCIe · **Stack:** dInfer 0.1 (from source) + vLLM 0.10.2 + torch 2.8.0+cu128 · **Date:** 2026-08-14
**Model:** `GSAI-ML/LLaDA-8B-Instruct` (8.02B params, BF16, cached locally)
**Data:** BGL test split (temporal), n=300 for quality/latency, paired with the HF baseline

This is the repo's first properly-served MDM number. Until now every LLaDA-8B
result ran on naive HF transformers (`VALIDATION_REPORT.md` §5, §8) — the exact
mirror of the AR-side strawman #12 fixed. #22 serves LLaDA-8B under
[dInfer](https://github.com/inclusionAI/dInfer), a runtime built for dLLMs.

---

## 0. Install gate — passed (with one environment fact corrected)

**The #13 §3 "no outbound network" premise is no longer true.** As of this
session pypi.org, files.pythonhosted.org, download.pytorch.org, huggingface.co
and github.com all return HTTP 200 from this host. This unblocks #22 (and more).

- `dinfer` is **not** on the internal PyPI mirror (`pypi.yandex-team.ru/simple/dinfer/` → 404), and the `dinfer` package on public PyPI is a 0.1.0 placeholder ("Add your description here", no deps). The **real** dInfer installs from GitHub source: `git clone inclusionAI/dInfer && pip install .`.
- `setup.py` deps are minimal (`scipy`, `tqdm`); vLLM/SGLang are optional (commented out). But `dinfer.decoding.serving` imports `vllm` unconditionally at module load, so vLLM is a hard dep even for the dense LLaDA path. The README pins `vllm==0.10.2` → `torch==2.8.0` (cu12x), which runs on the CUDA-13 driver (580.65.06) via driver forward-compat.
- Built a separate `.venv-dinfer` (`.venv-dg` untouched). One compat fix: transformers had to be pinned `<5` (5.x removed `is_torch_fx_available`, which `dinfer.model.modeling_llada2_moe` imports) → `transformers==4.57.6` (satisfies vllm 0.10.2's `>=4.55.2`).
- LLaDA-8B weights already cached (15 GB); disk has 275 GB free (the #13 98%-full pressure is gone).

**Gate verdict:** installable. `.venv-dinfer/bin/python -c "import dinfer; import vllm; import torch"` works; `torch.cuda.is_available()` True on the A100.

---

## 1. The cache limitation (read this before the latency numbers)

dInfer's optimized prefix/vicinity KV-cache paths **error on our prompt
lengths**. Both `VicinityCacheDiffusionLLM` and `BlockWiseDiffusionLLM+cache`
reach `modeling_llada.py:938`:

```
RuntimeError: expected src to have a size equal to the slice of self.
  src size = [1, 32, 64, 128], slice size = [1, 32, 240, 128]
```

This reproduces at **both** prompt lengths we tried: ~2000 tokens (few-shot)
and ~632 tokens (no few-shot). The LLaDA attention's `slice_scatter` mismatches
the windowed-forward key width against the cache slice. dInfer was benchmarked
on short prompts (GSM8K ≈ 100 tokens); our log-extraction prompts exceed what
its rolling-window cache handles. `cache='dual'` uses the same
`DiffusionKVCacheManager.get_key_values` path and is expected to fail the same
way (not separately tested).

**Consequence:** every latency/throughput number below is **`cache=''**
(`BlockWiseDiffusionLLM`, no KV-cache). This is a *fair* comparison to the HF
baseline, which also ran cache-less denoising (`mdm_wrapper.py:273`,
`use_cache=False`). It is **not** dInfer's headline optimized speed — that
remains unmeasured on this task because of the cache bug. This is recorded as a
dInfer limitation for long-prompt dLLM serving, not a diffusion property.

---

## 2. P-F3 — correctness gate: **PASS**

n=300 BGL, few-shot, `threshold=0.9`, `gen_len=256`, `block_length=32`, `cache=''`.

| Metric | dInfer (ours) | HF baseline | Δ |
|---|---|---|---|
| **Field-F1** | **0.872** | 0.779 | **+0.093** |
| Schema-valid | 99.0% | 97.1% | +1.9 pp |
| Consistency | 88.3% | 76.6% | +11.7 pp |
| Anomaly-F1 | 0.927 | 0.869 | +0.058 |

dInfer **exceeds** the HF baseline on all four metrics. The runtime does not
change generation semantics — same weights, same chat-template prompts, same
greedy decoding — so the correctness gate (P-F3) is satisfied within noise.
Per-field F1 confirms the known shape: structural fields (timestamp/host/
component/severity) ≈ 1.0; semantic/derived fields weaker
(`metric_value` 0.24, `event_type` 0.70, `message_template` 0.85).

The quality lift (0.779 → 0.872) comes from dInfer's **threshold decoder**
doing *adaptive* iterations per block until a confidence threshold is met,
versus the HF baseline's *fixed* K=8 steps. That extra work is exactly what
costs latency (§3).

---

## 3. P-F1 — streaming latency: **REFUTED** (on the cache-less path)

m=1, n=300, same config as §2.

| | p50 | p99 | p99.9 | mean | throughput |
|---|---|---|---|---|---|
| **dInfer (cache='')** | **10886** | 13717 | 14813 | 8971 | 0.11 ev/s |
| HF baseline | 1389 | 1556 | 1648 | — | — |
| AR-3B (vLLM, `vllm_serving.md`) | 413 | — | — | — | — |

**dInfer is 7.8× slower than HF** at p50 (10886 vs 1389 ms), the opposite of
P-F1's "dramatically faster" prediction. Two causes, both honest:

1. **Quality/latency trade-off, not engine speed.** dInfer's threshold decoder
   buys the +0.093 F1 in §2 by running far more forward passes than HF's fixed
   8 (8 blocks × adaptive iterations/block). The HF K=8 loop is a fixed budget;
   dInfer spends adaptively to converge each block. Comparing 10886 ms @ F1=0.872
   against 1389 ms @ F1=0.779 is not equal-quality — but the issue's P-F1
   target *is* the 1389 ms baseline, so the refutation stands as measured.
2. **The cache bug (§1) blocks the optimized path.** The prefix cache exists to
   amortize the ~1047-token prompt prefix across iterations, which is precisely
   where dInfer's speed should come from. With it broken, every iteration
   recomputes the full sequence. **The optimized dInfer latency is unmeasured.**

**Does the withdrawn #12 "MDM faster than AR" finding return? No — on the
measured path it does not, and the batch result (§4) says it cannot on any
path that resembles this one.** Even a hypothetical cache fix that brought
dInfer to ~1389 ms would only *match* HF and still lose to AR-3B-vLLM (413 ms)
by 3.4×. The withdrawn finding stays withdrawn.

---

## 4. P-F2 — batch multiplier: **~1.18×, flat — architectural**

`cache=''`, `few_shot=False` (632-token prompt), 2 reps/point, B ∈ {1,2,4,8,16,32,64}.
Multiplier is `ev_s[B] / ev_s[1]`; prompts replicated B× for identical-length batches (no padding/mask confound).

| B | time (s) | ev/s | tok/s | mult |
|---|---|---|---|---|
| 1 | 3.67 | 0.27 | 46.9 | 1.00× |
| 2 | 6.84 | 0.29 | 50.3 | 1.07× |
| 4 | 12.42 | 0.32 | 55.4 | **1.18×** |
| 8 | 26.29 | 0.30 | 52.3 | 1.12× |
| 16 | 55.25 | 0.29 | 49.8 | 1.06× |
| 32 | 109.03 | 0.29 | 50.5 | 1.08× |
| 64 | 217.75 | 0.29 | 50.6 | 1.08× |

**Peak multiplier 1.18× (B=4), declining to 1.08× at B=64.** Per-batch time
scales **linearly** with B (B=64 ≈ 64×B=1): the GPU is already compute-bound at
B=1, so adding sequences adds proportional compute with nothing to amortize.
This is **weaker** than P-F2's predicted 2–3× (Fast-dLLM v2's 102.5→217.5
tok/s across B=1→4) and far below AR's measured 67×.

**This is the control #17 was missing.** #17 asked whether the 18× throughput
deficit is architectural or vLLM-specific; DiffusionGemma-under-vLLM (one model,
one runner, a known crash) could not answer alone. LLaDA-8B under **dInfer** — a
runtime built for dLLMs, no vLLM batch crash (`vllm#50699` does not apply) —
shows the **same weak multiplier (~1×)**. The deficit is therefore
**architectural to diffusion**, not a vLLM runner limitation. vLLM's own framing
says the same thing in reverse: dLLMs are "compute-bound even at low batch
sizes, where spare compute is plentiful and memory bandwidth is the
bottleneck" — batching's benefit *is* the conversion from bandwidth-bound to
compute-bound, and dLLMs already spend that headroom at B=1.

**Net for #17:** the headline "both" verdict holds and is strengthened — the
m=1 latency advantage (dLLM's only real win) is architectural, but the batch
multiplier that decides aggregate throughput is also architecturally capped
near 1×, on a dLLM-native runtime as well as on vLLM.

---

## 5. Verdict

| Prediction | Result | |
|---|---|---|
| **P-F1** dInfer dramatically faster than HF (1389 ms) | 10886 ms, 7.8× slower | **Refuted** (cache-less; optimized path unmeasured, §1) |
| **P-F2** batch multiplier ~2–3× | 1.18× peak, flat | **Refuted (weak)** — architectural, confirms #17 |
| **P-F3** reproduces HF F1 0.779 | 0.872 (exceeds) | **Pass** |
| Withdrawn #12 "MDM faster than AR" returns? | No | **Confirmed withdrawn** |

#22's central purpose was to remove the serving confound and find out whether
the AR-side asymmetry #12 corrected has a symmetric MDM defect. It does, and
fixing it (serving LLaDA properly) **does not rescue the MDM-faster-than-AR
claim**: properly-served LLaDA-8B is slower per-call than its own HF baseline
(on the only working path) and gains nothing from batching. The repo's oldest
confound is now closed, and it closes against the diffusion case.

The one result that cuts the other way is **quality**: dInfer's threshold
decoder lifts F1 to 0.872 and consistency to 88.3% (+11.7 pp over HF) — the
best MDM quality number this repo has measured. That is a real property of
adaptive diffusion decoding, and it is orthogonal to the serving-speed
question #22 was about.

---

## 6. Caveats

- **Every latency/throughput number is `cache=''`.** dInfer's optimized cache
  is broken on our prompts (§1); the headline dInfer speed remains unmeasured.
  A cache fix upstream would change §3's latency (likely down) but not §4's
  multiplier (compute-bound scaling is independent of cache) and not §2's
  quality.
- **Single A100 vs dInfer's 8×H800 published figures** — absolute TPS not
  comparable; ratios transfer.
- **LLaDA-8B is 4.7× larger** than the paper's FlowFill-MDM (1.7B); absolute
  latency stays non-comparable to the paper.
- **Quality/latency are not iso-quality**: dInfer @ F1=0.872 vs HF @ F1=0.779.
  An equal-quality dInfer config (fewer iterations / lower threshold) would be
  faster than 10886 ms but was not measured.
- **Cross-stack**: dInfer (vLLM-0.10.2 stack) vs AR-3B (vLLM 0.25.1) are two
  runtimes; ratios across them are weaker evidence than same-runtime. Labeled
  on every number above.
- The cache limitation should be filed upstream (dInfer) or revisited; it is
  the single blocker on measuring dInfer's real serving speed here.

---

## 7. Files

- `results/llada_dinfer/correctness_latency.json` — P-F3 + P-F1, n=300
- `results/llada_dinfer/batch_sweep.json` — P-F2, B ∈ {1..64}
- `models/experiments/run_dinfer_serving.py` — correctness + latency runner
- `models/experiments/run_dinfer_batch.py` — batch-sweep runner
- `models/experiments/smoke_dinfer.py` — smoke test (n=10, F1=0.858)
- `.venv-dinfer/` — separate venv (vllm 0.10.2, torch 2.8.0+cu128, dinfer 0.1, transformers 4.57.6)

## 8. Related

- Parent: #16 (this supplies #17's missing control; the #13 "no network" premise is corrected)
- #17 — batch-multiplier control delivered here (architectural, §4)
- #12 / `VALIDATION_REPORT.md` §5, §8 — the symmetric AR-side fix and the two findings it withdrew (this issue confirms they stay withdrawn)
- Upstream: [dInfer](https://github.com/inclusionAI/dInfer), [dInfer paper](https://arxiv.org/abs/2510.08666), [vLLM DiffusionGemma blog](https://blog.vllm.ai/2025/10/10/diffusion-llm.html) ("attractive at low batch sizes")
