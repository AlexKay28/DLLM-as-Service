# DLLM-as-Service
DLLM as Service

Here's the meta, grouped as it appears in the paper.

**Models**

| Role | Name | Params | Family / basis | Purpose in paper |
|---|---|---|---|---|
| Ours | **FlowFill-MDM** | 1.7B | Masked diffusion LM, LLaDA/Dream recipe | The proposed engine; K=8 parallel denoising steps |
| Baseline (matched) | **AR-1.5B** | 1.5B | Qwen2.5-style autoregressive | Fair param-matched speed/accuracy baseline |
| Baseline (validity) | **AR-1.5B+grammar** | 1.5B | AR + XGrammar-constrained JSON | Shows constrained decoding buys validity, not speed |
| Ceiling | **AR-7B** | 7B | Qwen2.5-style autoregressive | Accuracy upper bound (esp. semantic slots) |

**Datasets / benchmarks**

| Corpus | Source scale | Sampled | Test anomaly % | Role |
|---|---|---|---|---|
| HDFS | 11.2M lines | 600k | 2.9 | Loghub — distributed FS logs |
| BGL | 4.7M lines | 600k | 7.3 | Loghub — Blue Gene/L, native alert labels |
| Thunderbird | 211M lines | 600k | 4.1 | Loghub — supercomputer logs |
| Spirit | 272M lines | 600k | 5.8 | Loghub — supercomputer logs |
| **Total** | — | **2.4M** | 5.0 | Temporal split: last 200k/corpus held out |

Label pipeline (not a benchmark, but part of the setup): **Drain** template parser for structural fields + a **teacher-distilled** subset for semantic fields; BGL/Thunderbird native alert categories give anomaly ground truth.

**Metrics**

| Metric | Axis | What it captures |
|---|---|---|
| Field-F1 | Quality | Token-level F1 per slot, macro-averaged |
| Schema-valid % | Quality | Parseable record, all keys present, correct types |
| Consistency % | Quality | Inter-field constraints hold (diffusion's weak spot) |
| Anomaly-F1 | Quality | Binary anomaly slot |
| p50 / p90 / p99 / **p99.9** | Latency | Tail is the alerting-SLA metric |
| Throughput @ 250ms p99 SLA | Serving | Events/s and tokens/s under fixed SLA |
| J / event | Efficiency | Board-level energy per record |

**Tricks / methods**

| Trick | What it does | Why it matters |
|---|---|---|
| L/T/S abstraction | Log batch / frozen Task / fixed Structure | Fixing S removes length-prediction + coherence problems |
| Fixed-schema infilling | Denoise only masked *values* of a template | Guarantees valid output; no grammar needed |
| Confidence-based unmasking | Commit high-confidence slots first, re-mask rest | +1.3 F1 at fixed K, no retraining |
| Frozen-T conditioning cache | Reuse Task-block representations across requests | Amortized in Packed mode; also credited to AR baselines (fair) |
| Semi-AR blocks (B) | Split S into B left-to-right blocks | Recovers cross-field consistency (B=2: 97.8→98.6%) |
| K denoising steps | Fixed step budget (K=8 knee) | O(K) content-independent latency → bounded tail |
| Streaming vs Packed modes | 1 event/context vs N/context | Min-latency vs max-throughput operating points |
| Iso-SLA framing | Compare under fixed p99, not iso-FLOP | The only axis diffusion honestly wins |
| PagedAttention + FlashAttention-2 serving | Standard efficient serving | Keeps the AR baselines strong (no strawman) |

Want this as a one-page `meta.md`, a slide, or an appendix table baked into the paper's LaTeX?
