# Evaluation Metrics

Metrics for evaluating and comparing models in the FlowFill paper.

---

## Overview

The paper defines metrics across three axes: **quality**, **latency**, and **serving efficiency**. It deliberately rejects the misleading "iso-FLOP" comparison and instead uses **iso-latency**, **iso-hardware**, and **iso-quality** axes.

| Axis | Metric | Type |
|---|---|---|
| Quality | Field-F1 | Per-slot extraction accuracy |
| Quality | Schema-valid % | Structural correctness |
| Quality | Consistency % | Inter-field constraint satisfaction |
| Quality | Anomaly-F1 | Anomaly detection quality |
| Latency | p50 / p90 / p99 / p99.9 | Latency percentiles |
| Serving | Throughput @ 250ms p99 SLA | Sustainable event rate |
| Serving | J / event | Energy efficiency |

---

## 1. Quality Metrics

### 1.1 Field-F1

**What:** Token-level F1 score per slot, macro-averaged across all 12 fields and 4 corpora.

**Formula:**
```
Field-F1 = (1/n) * Σ_{i=1}^{n} F1_i

where:
  F1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
  precision_i = TP_i / (TP_i + FP_i)
  recall_i    = TP_i / (TP_i + FN_i)
  n = 12 (number of schema slots)
```

- **TP:** Predicted token matches gold token at the correct position
- **FP:** Predicted token doesn't match gold at that position
- **FN:** Gold token not matched by prediction
- **Macro-average:** Each slot weighted equally, regardless of token count

**Paper values:**

| Model | Field-F1 |
|---|---|
| AR-1.5B | 0.913 |
| AR-1.5B+grammar | 0.916 |
| FlowFill-MDM (K=8) | 0.907 |
| AR-7B (ceiling) | 0.945 |

### 1.2 Schema-valid %

**What:** Percentage of records that are parseable as valid JSON with all required keys present and correct types.

**Formula:**
```
Schema-valid % = (count of valid records / total records) * 100

A record is valid if:
  1. Parseable as JSON
  2. All n=12 required keys present
  3. All values have correct types (string, float, bool, etc.)
```

**Paper values:**

| Model | Schema-valid % |
|---|---|
| AR-1.5B | 96.1 |
| AR-1.5B+grammar | 100.0 |
| FlowFill-MDM (K=8) | 99.4 |
| AR-7B (ceiling) | 97.4 |

**Note:** FlowFill achieves 99.4% without a grammar — the fixed template makes malformed output structurally impossible. Grammar-constrained AR reaches 100% but at a latency cost.

### 1.3 Consistency %

**What:** Percentage of records satisfying inter-field constraints.

**Formula:**
```
Consistency % = (count of consistent records / total records) * 100

Constraints checked (examples from paper):
  - severity ↔ is_anomaly  (ERROR → is_anomaly=true)
  - component ↔ error_code (error_code must be valid for the component)
  - severity ↔ anomaly_score (score must align with severity level)
```

**Paper values:**

| Model | Consistency % |
|---|---|
| AR-1.5B | 98.9 |
| AR-1.5B+grammar | 99.0 |
| FlowFill-MDM (K=8) | 97.8 |
| AR-7B (ceiling) | 99.4 |

**Note:** This is diffusion's weak spot — parallel decoding commits inter-dependent slots simultaneously. Semi-AR blocks (B=2) recover to 98.6%.

### 1.4 Anomaly-F1

**What:** Binary F1 score on the `is_anomaly` slot only.

**Formula:**
```
Anomaly-F1 = 2 * (precision * recall) / (precision + recall)

where:
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  TP = correctly predicted anomaly (is_anomaly=true and gold=true)
  FP = predicted anomaly but gold is normal
  FN = predicted normal but gold is anomaly
```

**Paper values:**

| Model | Anomaly-F1 |
|---|---|
| AR-1.5B | 0.871 |
| AR-1.5B+grammar | 0.874 |
| FlowFill-MDM (K=8) | 0.865 |
| AR-7B (ceiling) | 0.912 |

---

## 2. Latency Metrics

### 2.1 Latency Percentiles (p50, p90, p99, p99.9)

**What:** Wall-clock latency per event, measured at various percentiles.

| Percentile | Meaning | Importance |
|---|---|---|
| p50 | Median latency | Typical-case performance |
| p90 | 90th percentile | Sustained load behavior |
| p99 | 99th percentile | Standard SLA metric |
| **p99.9** | 99.9th percentile | **Alerting SLA metric** — decides if alerts fire in time |

**Key insight:** FlowFill's `O(K)` latency is content-independent (K=8 fixed steps), so p99.9 stays close to p50. AR's `O(|S|)` serial depth causes p99.9 to inflate on verbose/hard events.

**Paper values (Streaming mode, b=1):**

| Model | Steps | p50 ms | p99 ms | p99.9 ms |
|---|---|---|---|---|
| AR-1.5B | 128 | 168 | 240 | 340 |
| AR-1.5B+grammar | 128 | 205 | 292 | 410 |
| FlowFill-MDM (K=8) | 8 | 23 | 31 | 36 |
| AR-7B | 128 | 690 | 910 | 1240 |

**Bounded-latency property (Prop. 1):**
```
Λ_MDM = K * t_fwd(c + |S|)     # content-independent
Λ_AR  = Σ_{i=1}^{|S|} t_fwd(c + i) = Θ(|S| * t_fwd(c))  # serial, content-dependent
```

---

## 3. Serving & Efficiency Metrics

### 3.1 Throughput @ 250ms p99 SLA

**What:** Sustainable event rate under a fixed 250ms p99 latency SLA in Packed mode.

**Formula:**
```
Throughput = max_batch @SLA * (1000 / p50_ms)
           = events/s  (under the constraint that p99 <= 250ms)
```

- Batch size is tuned per model to the SLA
- Measures how many events/s a single GPU can serve without violating the tail

**Paper values (Packed mode, one A100):**

| Model | Max Batch @SLA | events/s | tokens/s |
|---|---|---|---|
| AR-1.5B | 24 | 640 | 82k |
| AR-1.5B+grammar | 20 | 520 | 67k |
| FlowFill-MDM (K=8) | 96 | 4,350 | 557k |
| AR-7B | 6 | 150 | 19k |

### 3.2 Energy per Event (J/event)

**What:** Board-level GPU power divided by events processed.

**Formula:**
```
J/event = (total_energy_J) / (num_events)
         = (avg_power_W * duration_s) / num_events
```

- Measured via board-level power integration on one A100

**Paper values (Streaming mode, b=1):**

| Model | J/event |
|---|---|
| AR-1.5B | 3.9 |
| AR-1.5B+grammar | 4.8 |
| FlowFill-MDM (K=8) | 0.7 |
| AR-7B | 22.5 |

---

## 4. Comparison Framing

The paper deliberately rejects **iso-FLOP** comparison (misleading — diffusion uses more raw FLOPs but converts idle bandwidth-bound cycles) and instead uses three axes:

| Axis | Principle | What it means |
|---|---|---|
| **Iso-latency** | Fix the SLA (e.g., 250ms p99), compare throughput | Who serves more events under the same deadline? |
| **Iso-hardware** | Same GPU, same software stack | No hardware advantage to either side |
| **Iso-quality** | Match Field-F1 within a band, compare latency/throughput | Who is faster at the same accuracy? |

### Headline Comparison

| Metric | FlowFill-MDM (K=8) | AR-1.5B | Win |
|---|---|---|---|
| Field-F1 | 0.907 | 0.913 | AR by 0.6 pts |
| Schema-valid % | 99.4 | 96.1 | FlowFill (+3.3) |
| Consistency % | 97.8 | 98.9 | AR by 1.1 pts |
| Anomaly-F1 | 0.865 | 0.871 | AR by 0.6 pts |
| p50 latency | 23 ms | 168 ms | **FlowFill 7.3x** |
| p99.9 latency | 36 ms | 340 ms | **FlowFill 9.4x** |
| Throughput @SLA | 4,350 ev/s | 640 ev/s | **FlowFill 6.8x** |
| Energy/event | 0.7 J | 3.9 J | **FlowFill 5.6x** |

---

## 5. Semi-AR Block Ablation Metrics

When using B blocks (left-to-right, each with K/B diffusion steps):

| Variant | Field-F1 | Consist. % | p50 ms | vs. AR |
|---|---|---|---|---|
| B=1 (parallel) | 0.907 | 97.8 | 23 | 7.3x |
| B=2 | 0.912 | 98.6 | 29 | 5.8x |
| B=4 | 0.914 | 99.0 | 41 | 4.1x |

---

## References

- Paper §5 (Experimental Setup — Metrics)
- Paper §6 (Results — Tables 2, 3, 4)
- Paper §7 (Analysis — Table 5, Figures 2-4)
