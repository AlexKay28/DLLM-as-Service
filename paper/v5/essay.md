# Paper Structure — v4

**Title:** Diffusion LLMs as Fast Models for Specific Tasks: The Cost Shape of Block-Structured Decoding

**Abstract.** AR decoding slows per token; block-diffusion has a staircase cost (flat inside a canvas, stepping at boundaries). On a pinned vLLM stack with MoE-matched arms on structured log extraction, shapes cross at short output length. Above it, block diffusion is 2.5× faster at natural output, narrowing to ~parity at 1000 tokens. AR wins under high throughput (better batching, tighter tail). Block diffusion wins where requests arrive serially (investigation chains, autonomous driving, trading). Under a tightening deadline, block diffusion degrades quality rather than truncating. Five conditions define the regime where it is the right choice.

---

## Main body

### §1 Introduction
- Framing: AR = metered taxi (fare climbs per token); block-diffusion = subway (flat fare per station).
- Both run on vLLM on one GPU (A100), same serving stack, not identical engine versions.
- Task: structured log extraction — raw log line → JSON record with 12 fields (structural / semantic / anomaly / action).
- Data: BGL from LOGHUB, 4.7M lines; 600k sampled, temporal split (200k test / 400k train). Anomaly rate 20.5%.
- Quality: macro token-F1 over 9 of 12 fields (3 excluded = parser placeholders). Schema-validity = JSON parses + 12 keys. n=300 per cell.
- Arms (both MoE): DG = DiffusionGemma-26B-A4B (4B active of 26B); AR-30B = Qwen3-30B-A3B (3B active of 30B). Matched on architecture class and active params.
- **Plot twist:** initial 4× DG win was a harness defect (AR on naive HF loop, DG on vLLM). Moving AR onto same stack erased the gap. Eight defects found by re-reading artifacts; five biased against AR. Reported advantage survived all nine corrections *and* the matched-baseline switch.
- Contributions: (a) two cost laws with crossover $L^{*}$; (b) where DG is faster; (c) what limits it operationally; (d) five conditions for the right choice.
- Table 1: paper-in-one-table (matched arms, marginal cost, batch multiplier, tail). Table 2: the two cost laws side-by-side.

### §2 Background: AR vs block-diffusion decoding
- How each decodes: AR one token per forward pass; DG a canvas of C tokens via K denoising passes, free inside a canvas, cost jumps at boundaries.
- **Cost laws** (Eq. 1, 2): $T_{AR}=a+bL$ (linear); $T_{DG}=a+b(\lceil L/C\rceil-1)$ (staircase).
- AR intercept = prefill (amortised by prefix cache); slope = one KV-cache pass per token. DG intercept = first-canvas cost (prompt re-attended K times); slope = one extra canvas.
- Operating point K=8 = latency/quality knee (quality flat above 8, latency saturates above 16).
- **Crossover** (Eq. 3): $L^{*}=(a_{DG}-a_{AR})/b_{AR}=48$ tokens; $\lambda^{*}=1000/a_{DG}$ events/s. Below $L^{*}$ AR cheaper; above, DG, non-monotonic (climbs inside canvas, drops at edges).
- Fig 2: staircase probed at canvas boundaries — steps regular (8% spread, median canvas cost), so slope is one constant.
- Base decomposition: K=2→8 interpolation splits base into 143ms floor (38%) + 236ms per-step (62%). Per-step larger because prefill paid K times; caching prompt KV across steps could collapse most of 62%.
- **Generalisation:** BGL→HDFS within 2% (canvas step 1.7% apart at K=8). Shape is a model property, not data. Caveats: HDFS is small sample; AR generalisation on dense AR-3B not AR-30B.
- Shape transfers across tasks: HumanEval code-gen, AR 6.4ms/token vs 7.2 on logs, DG flat 280ms staircase, ratio 2.9× (vs 2.5× on logs).

### §3 Results
- Serial mode: AR-30B ~1s for a 126-token record; DG <400ms. AR stops at EOS so plateau regardless of max_tokens.
- Table 3 (serial latency): AR flat from 128 tokens (EOS at ~126); DG flat first canvas then steps. Ratio: 0.7× (truncated AR at 32) → 2.5× (natural) → 1.3× (1024). Falls further past table (0.8× at 2048). DG advantage holds at typical lengths, not universally.
- **DG wins at every practical output length** — serial crossover at $L^{*}=48$ ($L\geq48$ integer-safe). AR serial slope is 23× steeper than its batch-throughput slope; quoting batch slope as latency understates AR serial by 23×.
- **Energy:** DG draws 10.8W net vs AR-30B's 173.3W. 32kWh gap per million events. $R_{energy}=118.7/4.2=28.3$ (AR-30B serial); dense AR-1.5B ratio only 1.2. Both runs serial (matching DG regime).
- **Graceful degradation:** under deadline, DG returns complete canvas at lower quality (drop K 8→4, F1 falls, p99 budget met); AR returns truncated record failing schema. K is a per-request quality knob.
- Table 4 (challenge contrast): long output / deadline / arrival rate / tail SLA / energy / long prompt — each row names the architectural driver. Tail stats are dense AR-3B (only arm measured per-event); AR-30B MoE tail unmeasured.

### §4 What limits the advantage operationally
Five deployment factors:
1. **Footprint** — both MoE comparable; neither shares GPU. Real asymmetry vs dense AR-1.5B at 3GB (16 copies fit).
2. **Batching** — continuous batching 25× for AR-30B vs ≤1.7× for DG. Both DG multipliers engine-bounded (true ceiling unknown but ≥1.7). >order of magnitude gap. Fig 6: DG caps on vLLM, LLaDA on dInfer — two families, same ceiling (changing attention masks invalidate KV persistence). Prefix cache on/off doesn't create the gap.
3. **Tail latency** — DG CV and max/median 3.21× vs dense AR-3B tight. 3.21× tail breaks a batch-1 SLA — exactly DG's per-event regime.
4. **Input length** — DG prefill slope 2.6× steeper than dense AR-3B (each denoising step re-attends full prompt). Longer inputs push $L^{*}$ upward.
5. **Co-residency** — if batch-1 is tenant isolation, 16× AR-1.5B copies in same 52GB; 16× throughput, isolation preserved. Not measured (capacity argument).
- **Cost per event** (Table 5, A100 $4.62/hr): DG $0.0005 vs AR-30B $0.0012 (100% duty). Against dense AR-1.5B single copy $0.0009 → ratio 1.8×; with 16 co-resident copies projected $9× cheaper. Takeaway: batch → AR wins economics; can't batch → DG wins latency.

### §5 What survives
- Five conditions for DG regime: (1) output ≥48 tokens; (2) batch-1 by requirement; (3) completeness validated against schema; (4) quality floor exceeds AR's; (5) 52GB footprint acceptable.
- **Serial investigation chains (SOC):** each hop blocks previous, concurrency 1. DG 419ms/hop (re-prefills longer prompt) vs AR-30B 945ms anchor; $2.3×$ is cross-harness (only dense AR-3B run through chain, 808ms/hop → 1.9× in figure). Shape: neither per-hop cost nor DG tail grows with depth. 8 hops: 3s vs 8s.
- HumanEval multi-turn code chain confirms: DG p50 flat 287–305ms (+6%), AR grows 511→620ms (+21%).
- **Deadline expiry mid-generation:** preempted DG returns valid record; AR returns half-JSON. K knob = emergency brake (8→4 still fills fields). Completeness guarantee only above canvas length; below L=128 DG schema validity collapses.
- **Autonomous driving** = batch-1 by construction (one vehicle, one scene). Long output (scene, trajectories) = where staircase advantage largest.
- **Algorithmic trading** = serial chain, 1–5s reaction window; taxi fare compounds, subway doesn't.
- **Wide schemas** favour joint denoising: 12→100 fields, DG advantage 1.8×→3.0× (vs dense AR-3B). Stops growing at 100 (output exceeds one canvas).
- **Anytime escalation:** K-cascade (K=2 first, escalate incomplete to K=8). AR has no equivalent (binary EOS). Measured on BGL: 29.5% resolve at K=2, rest escalate; cascade F1 0.848 (−0.005) but avg latency 465ms > fixed K=8 376ms (K=2 overhead not amortised when 70% need K=8). Structural property confirmed; practical benefit task-dependent.
- **Where diffusion doesn't win:** high-rate streams (AR batching), tight p99 SLAs (DG tail), anything that can batch.
- **Where diffusion fails:** six hypotheses refuted (Appendix D). Three run on LLaDA-8B vs dense AR-3B (predate DG arm). Bidirectional attention propagates noise both ways (DG *less* robust in all 10 categories). Joint denoising should preserve suffix but AR regenerates trivially. Selective re-masking gains small F1 over 3 rounds but still trails AR after one pass. Block diffusion's advantage is latency shape at long output, not robustness/infilling/refinement.

### §6 Related work
- LLaDA, DiffusionGemma claim speedups vs author-built AR baselines; this paper pins both to one stack and fits cost laws (shape vs point).
- Constrained decoding (Outlines, xgrammar, llguidance) — not measured, flagged in threats.
- Speculative decoding (Medusa, EAGLE) attacks the AR slope but keeps linear shape. Measured n-gram prompt-lookup on dense AR-3B: p50 −1/3 but p99.9 3× (rejected speculations re-decode). AR tail still 5× worse than DG worst-case p99.9. Staircase wins on tail; speculation narrows median, not structure.
- Hoefler & Belli reproducibility pitfalls — the nine corrections are a case study in that tradition.

### §7 Threats to validity
- One GPU; constants A100-specific, shapes should transfer. Crossover calibrated at ~reference prompt, shifts with input length.
- **Quality equivalence** ±0.015 ($\Delta$=qualityGap, 95% CI [−0.012,+0.011], n=300). Looser than pre-registered ±0.010; at tighter margin 95% CI doesn't fit, equivalence only on 90% CI.
- Quality arm = dense AR-3B, not AR-30B (AR-30B 9-field F1 0.870, slightly above dense). Latency arm = AR-30B. **Quality and latency arms are not the same model.**
- Secondary results from other arms flagged: per-event tail = dense AR-3B (AR-30B never run per-event); BGL→HDFS AR side = dense AR-3B; perturbation/refinement/bounded-deadline = LLaDA vs dense AR-3B; serial headline is batch-1 (same AR-30B batches 25×, invert economics).
- Niche untested on task with natural output ≥512 tokens. AR-30B cost law fitted 5-point sweep $R^2=0.9996$ but model stops at ~126 tokens (EOS), slope carried by L=32 + natural cluster. Forcing AR longer would widen DG advantage (Appendix H).
- Niche untested vs grammar-constrained AR (would eliminate over-generation, 100% schema validity, remove completeness advantage). Attempt on dense AR-1.5B (vLLM 0.10.2, outlines) → 1.0% schema-valid (grammar enforced JSON but model padded unanswerable strings). One model, one older engine — bounds nothing.
- Batching ceiling reproduces across two runtimes/families but both engine-bounded; architectural argument in Appendix I.
- Cost-shape generalisation from one diffusion model vs one AR family; shape structural but not tested on second diffusion family.
- Energy comparison uses three different idle baselines; DG energy not re-measured at AR-30B idle point — ratio indicative.
- **Tail re-measured at n=2000 (issue #120):** DG p99.9 557ms vs max 1166ms; max/med 3.07× (was 3.21× at n=300); CV 16.9% (was 20.9%). AR-3B tight as before. Bootstrap 95% CI for DG p99.9 [493, 1166]ms — tail wide, point estimate distinct from max, ratio stable across sample sizes.
- K-cascade measured on BGL; doesn't save latency here (70% escalation) but structural property confirmed; benefit task-dependent.
- DG checkpoint: DiffusionGemma-26B-A4B-it on HuggingFace.

### §8 Conclusion
- Staircase vs linear; crossover $L^{*}=48$. Advantage moderate: 2.5× at natural output, ~parity at 1000 tokens. In cost: $1200/M events, $2600/year on one GPU at 10k/day. Bigger AR doesn't close it; batching and co-residency do.
- Matched baseline adds only 15% to serial per-token cost; crossover stays at 48 (close to dense AR-3B's 59). DG wins every practical serial length, margin narrower than earlier draft.
- Advantage limited where batching available and tail SLAs tight. In batch-1 (SOC, driving, trading) staircase is structural. K knob = emergency brake (graceful vs truncation). Five conditions named and grounded.

---

## Appendices

- **A. Defects** (`09_appendix_defects.tex`) — the nine priced corrections; methodological backbone.
- **B. Decomposition** (`10_appendix_decomp.tex`) — base split into K-independent floor (~143ms) and per-step work (~236ms).
- **C. Batching** (`11_appendix_batching.tex`) — AR 25× vs DG ≤1.7×, both engine-bounded.
- **D. Refuted** (`12_appendix_refuted.tex`) — six refuted hypotheses; three on LLaDA vs dense AR-3B.
- **E. Input canvas** (`13_appendix_input_canvas.tex`) — DG prefill slope 2.6× steeper; canvas setting that improves latency and quality at once.
- **F. Quality** (`14_appendix_quality.tex`) — schema validity below L=128 collapses; F1 details.
- **G. Distributions** (`15_appendix_distributions.tex`) — per-event tail; n=2000 re-measurement.
- **H. Forced** (`16_appendix_forced.tex`) — projection through multi-record task with long natural output.
- **I. Batching architectural** (`17_appendix_batching_arch.tex`) — structural argument for why batching ceiling is architectural (attention masks invalidate KV persistence).
- **J. HumanEval** (`18_appendix_humaneval.tex`) — code-gen second task; multi-turn chain (generate→execute→fix).

---

## Notable honesty flags
- Per-event tail statistics use dense AR-3B, not AR-30B (AR-30B never run per-event).
- Quality arm ≠ latency arm.
- Several probes used LLaDA-8B, not DG.
- Initial 4× DG win was a harness defect; reported advantage survived nine corrections.
- Quality equivalence ±0.015 looser than pre-registered ±0.010.
- K-cascade doesn't save latency on BGL (70% escalation); structural property only.
