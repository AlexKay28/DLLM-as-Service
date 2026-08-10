# Models

Models evaluated, compared, and discussed in the FlowFill paper.

---

## 1. Models in the Paper

The paper compares **4 LLM-based models**. No rule-based or classical methods are included as baselines.

| Role | Name | Params | Family / Basis | Purpose |
|---|---|---|---|---|
| **Ours** | FlowFill-MDM | 1.7B | Masked diffusion LM (LLaDA/Dream recipe) | The proposed engine; K=8 parallel denoising steps |
| Baseline (matched) | AR-1.5B | 1.5B | Qwen2.5-style autoregressive | Fair param-matched speed/accuracy baseline |
| Baseline (validity) | AR-1.5B+grammar | 1.5B | AR + XGrammar-constrained JSON | Shows constrained decoding buys validity, not speed |
| Ceiling | AR-7B | 7B | Qwen2.5-style autoregressive | Accuracy upper bound (esp. semantic slots) |

### 1.1 FlowFill-MDM (Proposed)

- **Architecture:** Absorbing-state masked diffusion language model
- **Family:** LLaDA / Dream recipe
- **Params:** 1.7B
- **Key mechanism:** Denoises all 12 schema slots in K=8 parallel steps (independent of content)
- **Confidence-based unmasking:** High-confidence slots committed first, rest re-masked
- **Semi-AR blocks (optional):** Split S into B left-to-right blocks for cross-field consistency
- **Frozen-T cache:** Task block representations computed once, reused across requests
- **Operating modes:** Streaming (m=1, min latency) / Packed (m>1, max throughput)

### 1.2 AR-1.5B (Param-Matched Baseline)

- **Architecture:** Standard autoregressive transformer
- **Family:** Qwen2.5-style
- **Params:** 1.5B
- **Decoding:** Serial, 128 forward passes per record (one token at a time)
- **Serving:** PagedAttention + FlashAttention-2 (no strawman)

### 1.3 AR-1.5B+grammar (Validity Baseline)

- **Same as AR-1.5B** but with XGrammar-constrained JSON decoding
- **Purpose:** Shows that grammar constraints buy schema-validity (100%), not speed (actually slower: 205ms vs 168ms p50)
- **Grammar overhead:** Mask computation adds latency on every step

### 1.4 AR-7B (Accuracy Ceiling)

- **Architecture:** Standard autoregressive transformer
- **Family:** Qwen2.5-style
- **Params:** 7B
- **Purpose:** Accuracy upper bound — especially on semantic slots (event_type, affected_entity, remediation_hint)
- **Cost:** 690ms p50, 22.5 J/event — too slow for real-time alerting

---

## 2. Shared Configuration

All models share:
- **Tokenizer:** Same across all models
- **Training targets:** Same distilled targets (Drain + teacher distillation)
- **Prompt:** Same input format
- **Hardware:** One NVIDIA A100-80GB (PCIe), bf16, CUDA 12.4
- **Serving stack:** FlashAttention-2, PagedAttention
- **Batch size:** Tuned per model to the SLA

---

## 3. Missing Baselines — Rule-Based Methods

The paper does **NOT** include any rule-based or classical baselines. This is a notable gap.

### 3.1 What's Missing

| Method | Type | What It Does | Included? |
|---|---|---|---|
| **grep / awk / sed** | Unix tools | Pattern matching, field extraction from raw logs | No |
| **Regex rules** | Rule-based | Per-source parser patterns (the "brittle" approach the paper dismisses in §1) | No |
| **Drain (parser only)** | Log parsing | Template extraction → structured events without any LLM | No (used only for labeling) |
| **Logstash / Fluentd** | Log pipeline | Config-based pipeline with grok patterns | No |
| **DeepLog** | Deep learning | LSTM-based anomaly detection from log sequences | No (cited as ref [5] but not benchmarked) |
| **LogBERT** | Pre-trained model | BERT-based log anomaly detection | No (cited as ref [7] but not benchmarked) |

### 3.2 Why This Matters

The paper's §1 states: *"a single prompt replaces brittle per-source regexes and parser rules"* — but never actually benchmarks against them. A fair comparison would show:

| Axis | grep/awk/regex | Drain (parser only) | LLM (AR or FlowFill) |
|---|---|---|---|
| **Speed** | Microseconds, CPU-only | Milliseconds, CPU-only | 23-690ms, GPU required |
| **Structural fields** | High (exact match if patterns well-tuned) | High (template extraction) | High |
| **Semantic fields** | Zero (can't classify event_type, generate remediation_hint) | Zero (templates only) | Moderate-high |
| **Anomaly detection** | Binary pattern match only | Template-based, no semantic | Moderate (0.865-0.912 F1) |
| **Schema-validity** | 100% (you define output format) | 100% (structured CSV) | 96-100% |
| **Cross-field consistency** | 100% (rules enforce it) | 100% (deterministic) | 97.8-99.4% |
| **Maintainability** | Rules break when log format changes | Adapts via re-training on new templates | Generalizes to format shifts |
| **New schema field** | Write new regex | Re-run Drain + manual mapping | Update prompt, re-fine-tune |
| **Hardware cost** | $0 (CPU) | $0 (CPU) | A100 GPU ($$$) |

### 3.3 The Honest Picture

Rule-based methods win on:
- **Speed** (orders of magnitude faster)
- **Cost** (CPU-only, no GPU)
- **Structural field accuracy** (exact match)
- **Schema-validity and consistency** (deterministic by construction)

LLMs win on:
- **Semantic fields** (event_type classification, remediation_hint generation)
- **Generalization** (handle unseen log formats without new rules)
- **Flexibility** (add a field by updating the prompt, not writing regex)

**The real question the paper doesn't answer:** For a 12-field schema where ~8 fields are structural (extractive), would a hybrid approach (grep/Drain for structural + small LLM for semantic only) be faster, cheaper, and nearly as accurate?

### 3.4 Classical Methods Referenced (but not benchmarked)

| Method | Paper Ref | What | Why Not Benchmarked |
|---|---|---|---|
| Drain | [8] | Online log parsing, fixed-depth tree | Used only for label generation, not as a baseline |
| DeepLog | [5] | LSTM anomaly detection from log sequences | Cited but not compared |
| LogBERT | [7] | BERT-based log anomaly detection | Cited but not compared |
| Outlines | [20] | Grammar-guided generation | Cited as related work |
| XGrammar | [4] | Constrained JSON decoding | Used as a baseline (AR-1.5B+grammar) |

---

## 4. Results Comparison

### 4.1 Quality (Table 2)

| Model | Field-F1 | Schema-valid % | Consist. % | Anomaly-F1 |
|---|---|---|---|---|
| AR-1.5B | 0.913 | 96.1 | 98.9 | 0.871 |
| AR-1.5B+grammar | 0.916 | 100.0 | 99.0 | 0.874 |
| **FlowFill-MDM (K=8)** | 0.907 | 99.4 | 97.8 | 0.865 |
| AR-7B (ceiling) | 0.945 | 97.4 | 99.4 | 0.912 |

### 4.2 Streaming Latency & Energy (Table 3)

| Model | Steps | p50 ms | p99 ms | p99.9 ms | J/event |
|---|---|---|---|---|---|
| AR-1.5B | 128 | 168 | 240 | 340 | 3.9 |
| AR-1.5B+grammar | 128 | 205 | 292 | 410 | 4.8 |
| **FlowFill-MDM (K=8)** | 8 | 23 | 31 | 36 | 0.7 |
| AR-7B | 128 | 690 | 910 | 1240 | 22.5 |

### 4.3 Packed-Mode Throughput @ 250ms p99 SLA (Table 4)

| Model | Max Batch @SLA | events/s | tokens/s |
|---|---|---|---|
| AR-1.5B | 24 | 640 | 82k |
| AR-1.5B+grammar | 20 | 520 | 67k |
| **FlowFill-MDM (K=8)** | 96 | 4,350 | 557k |
| AR-7B | 6 | 150 | 19k |

### 4.4 Semi-AR Block Ablation (Table 5)

| Variant | Field-F1 | Consist. % | p50 ms | vs. AR |
|---|---|---|---|---|
| B=1 (parallel) | 0.907 | 97.8 | 23 | 7.3x |
| B=2 | 0.912 | 98.6 | 29 | 5.8x |
| B=4 | 0.914 | 99.0 | 41 | 4.1x |

---

## 5. Per-Field Breakdown (Fig. 4)

| Field | Category | FlowFill vs AR-1.5B | Notes |
|---|---|---|---|
| timestamp_norm | Structural | Match | Extractive — copy/normalize |
| host | Structural | Match | Extractive |
| severity | Structural | Match | Extractive — enum classification |
| component | Structural | Match | Extractive |
| message_template | Structural | Match | Drain-style template |
| error_code | Structural | Match | Extractive |
| metric_value | Structural | Match | Extractive — numeric |
| event_type | Semantic | FlowFill trails | Requires inference/classification |
| affected_entity | Semantic | FlowFill trails | Requires inference |
| is_anomaly | Anomaly | Match | Binary, largely extractive |
| anomaly_score | Anomaly | Slight gap | Continuous, depends on context |
| remediation_hint | Semantic | Largest gap | Free-form generation, AR-7B helps most |

**Key insight:** Diffusion is essentially free where the task is copy/normalize (structural slots), and pays only where the task is inference (semantic slots).

---

## 6. What's NOT Available

The paper is a **preprint under review** — the following are NOT released:

- FlowFill-MDM model weights (1.7B)
- AR baseline model weights (1.5B, 7B)
- Training scripts (fine-tuning recipes)
- Serving infrastructure code (Streaming/Packed modes, frozen-T cache)
- Prompt templates
- Semi-AR block implementation

All model names are anonymized ("Qwen2.5-style") and no checkpoints or code repos are linked.

---

## 7. Proposed Additional Baselines

For a more complete evaluation, the following baselines should be considered:

| Baseline | Type | Why It Matters |
|---|---|---|
| **grep/awk extractor** | Rule-based | Floor of effort — fastest possible, zero GPU |
| **Drain-only (no LLM)** | Log parser | Shows what template parsing alone achieves |
| **Hybrid: Drain + small classifier** | Pipeline | Structural fields via Drain, semantic via small model |
| **DeepLog** | Deep learning | LSTM anomaly detection — classical ML baseline |
| **LogBERT** | Pre-trained | BERT-based — pre-LLM deep learning baseline |
| **Speculative decoding AR** | Fast AR | Reduces AR constant factor, preserves serial depth |

---

## References

- Paper §1 (Introduction — dismisses rule-based methods)
- Paper §2 (Related Work — cites DeepLog, LogBERT, Drain, but doesn't benchmark them)
- Paper §5 (Experimental Setup — Models)
- Paper §6 (Results — Tables 2-4)
- Paper §7 (Analysis — Table 5, Fig. 4)
