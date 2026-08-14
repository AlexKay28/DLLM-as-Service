# Issue #24: Adaptive-effort via threshold decoding

## Per-bin results

| Model | Bin | N | F1 | p50ms | p99ms | mean_ms |
|---|---|---|---|---|---|---|
| AR-3B (vLLM) | routine | 200 | 0.855 | 826 | 1374 | 874 |
| AR-3B (vLLM) | medium | 200 | 0.833 | 808 | 1299 | 827 |
| AR-3B (vLLM) | novel | 200 | 0.788 | 807 | 1624 | 823 |

| dInfer (threshold) | routine | 100 | 0.873 | 10823 | 14645 | 9249 |
| dInfer (threshold) | medium | 100 | 0.843 | 10995 | 14815 | 8693 |
| dInfer (threshold) | novel | 100 | 0.886 | 11009 | 13433 | 8868 |

## Verdicts

- **P-F1** (DM F1 > AR F1 on novel): PASS (DM=0.886, AR=0.788)
- **P-F2** (DM p50 < AR p50 on routine): REFUTED (DM=10823ms, AR=826ms)
- **P-F3** (DM latency range > AR across bins): PASS (DM=557ms, AR=51ms)
- **P-F4** (DM F1 drop routine->novel < AR): PASS (DM=-0.013, AR=0.067)
