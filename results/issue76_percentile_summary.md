# Issue #76: Percentile Summary

Recomputed p50/p95/p99 for 530 latency cells from 130 files.
Cells with raw data: 6. Summary-only: 524.

## AR-1.5B serving

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256 | 2000 | 413 | - | 685 | 806 | 1.95x |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=1, latency_source=burst_wall_clock | 2000 | - | - | 419 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=2, latency_source=burst_wall_clock | 2000 | - | - | 449 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=4, latency_source=burst_wall_clock | 2000 | - | - | 470 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=8, latency_source=burst_wall_clock | 2000 | - | - | 511 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=16, latency_source=burst_wall_clock | 2000 | - | - | 538 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=32, latency_source=burst_wall_clock | 2000 | - | - | 634 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=64, latency_source=burst_wall_clock | 2000 | - | - | 970 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=128, latency_source=burst_wall_clock | 2000 | - | - | 1141 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=256, latency_source=burst_wall_clock | 2000 | - | - | 1562 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=512, latency_source=burst_wall_clock | 2000 | - | - | 2499 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256 | 500 | 426 | - | 818 | 824 | 1.94x |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=1, latency_source=burst_wall_clock | 500 | - | - | 444 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=2, latency_source=burst_wall_clock | 500 | - | - | 485 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=4, latency_source=burst_wall_clock | 500 | - | - | 524 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=8, latency_source=burst_wall_clock | 500 | - | - | 595 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=16, latency_source=burst_wall_clock | 500 | - | - | 690 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=32, latency_source=burst_wall_clock | 500 | - | - | 913 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=64, latency_source=burst_wall_clock | 500 | - | - | 1546 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=128, latency_source=burst_wall_clock | 500 | - | - | 2304 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256, batch_size=256, latency_source=burst_wall_clock | 500 | - | - | 3876 | - | - |

## AR-1.5B variants

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256 | 300 | 666 | - | 1057 | - | - |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=1024, prompt_token_count=992 | 100 | 4026 | - | 4051 | 4051 | 1.01x |
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=512, prompt_token_count=992 | 100 | 2005 | - | 2026 | 2026 | 1.01x |
| Qwen/Qwen2.5-1.5B-Instruct | direct | 100 | 1007 | - | 1039 | - | - |

## AR-3B HDFS

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct* | max_new_tokens=256, prompt_token_count=947 | 300 | 971 | 1250 | 1398 | 1406 | 1.45x |
| Qwen/Qwen2.5-3B-Instruct* | max_new_tokens=256, prompt_token_count=947 | 300 | 1918 | 1929 | 1936 | 1979 | 1.04x |

## AR-3B serving

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256 | 500 | 807 | - | 1300 | 1625 | 2.01x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=1, latency_source=burst_wall_clock | 500 | - | - | 849 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=2, latency_source=burst_wall_clock | 500 | - | - | 808 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=4, latency_source=burst_wall_clock | 500 | - | - | 871 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=8, latency_source=burst_wall_clock | 500 | - | - | 896 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=16, latency_source=burst_wall_clock | 500 | - | - | 1067 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=32, latency_source=burst_wall_clock | 500 | - | - | 1402 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=64, latency_source=burst_wall_clock | 500 | - | - | 1706 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=128, latency_source=burst_wall_clock | 500 | - | - | 2453 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=256, latency_source=burst_wall_clock | 500 | - | - | 3242 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256 | 2000 | 662 | - | 1062 | 1336 | 2.02x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=1, latency_source=burst_wall_clock | 2000 | - | - | 695 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=2, latency_source=burst_wall_clock | 2000 | - | - | 683 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=4, latency_source=burst_wall_clock | 2000 | - | - | 720 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=8, latency_source=burst_wall_clock | 2000 | - | - | 757 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=16, latency_source=burst_wall_clock | 2000 | - | - | 903 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=32, latency_source=burst_wall_clock | 2000 | - | - | 1187 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=64, latency_source=burst_wall_clock | 2000 | - | - | 1378 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=128, latency_source=burst_wall_clock | 2000 | - | - | 1789 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=256, latency_source=burst_wall_clock | 2000 | - | - | 2189 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, batch_size=512, latency_source=burst_wall_clock | 2000 | - | - | 3527 | - | - |

## AR-3B variants

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct* | max_new_tokens=256, prompt_token_count=992 | 300 | 1915 | 1923 | 1929 | 1941 | 1.02x |
| Qwen/Qwen2.5-3B-Instruct* | max_new_tokens=256, prompt_token_count=992 | 300 | 1264 | 2096 | 2388 | 5167 | 4.98x |
| Qwen/Qwen2.5-3B-Instruct* | max_new_tokens=256, prompt_token_count=992 | 300 | 950 | 1159 | 1512 | 1608 | 1.69x |
| Qwen/Qwen2.5-3B-Instruct* | max_new_tokens=256, prompt_token_count=992 | 300 | 1810 | 2492 | 3667 | 9909 | 6.88x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256 | 300 | 1018 | - | 1612 | - | - |

## AR-7B variants

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256 | 300 | 1530 | - | 2034 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256 | 2000 | 1161 | - | 1521 | 2311 | 1.99x |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=1, latency_source=burst_wall_clock | 2000 | - | - | 1199 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=2, latency_source=burst_wall_clock | 2000 | - | - | 1179 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=4, latency_source=burst_wall_clock | 2000 | - | - | 1220 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=8, latency_source=burst_wall_clock | 2000 | - | - | 1297 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=16, latency_source=burst_wall_clock | 2000 | - | - | 1517 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=32, latency_source=burst_wall_clock | 2000 | - | - | 1713 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=64, latency_source=burst_wall_clock | 2000 | - | - | 2278 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=128, latency_source=burst_wall_clock | 2000 | - | - | 2361 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=256, latency_source=burst_wall_clock | 2000 | - | - | 3561 | - | - |
| Qwen/Qwen2.5-7B-Instruct | max_new_tokens=256, batch_size=512, latency_source=burst_wall_clock | 2000 | - | - | 6015 | - | - |

## Adaptive effort

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| LLaDA-8B-Instruct | medium | 100 | 10995 | - | 14815 | - | - |
| LLaDA-8B-Instruct | novel | 100 | 11009 | - | 13433 | - | - |
| LLaDA-8B-Instruct | routine | 100 | 10823 | - | 14645 | - | - |
| Qwen/Qwen2.5-3B-Instruct | medium | 200 | 808 | - | 1299 | - | - |
| Qwen/Qwen2.5-3B-Instruct | novel | 200 | 807 | - | 1624 | - | - |
| Qwen/Qwen2.5-3B-Instruct | routine | 200 | 826 | - | 1374 | - | - |

## Baselines

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| ? | direct | - | 0.05 | - | 0.11 | 37.1 | - |

## Canvas sweep (DG)

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| google/diffusiongemma-26B-A4B-it | canvas_length=128 | 200 | 368 | - | 628 | - | - |
| google/diffusiongemma-26B-A4B-it | canvas_length=256 | 200 | 376 | - | 494 | - | - |
| google/diffusiongemma-26B-A4B-it | canvas_length=64 | 200 | 557 | - | 769 | - | - |

## Chain analysis

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| ? | mixed_1dec_4ext_ar3b | - | 3463 | - | 3662 | 3727 | - |
| ? | mixed_1dec_4ext_dg | - | 2143 | - | 2826 | 3085 | - |
| ? | mixed_1dec_4ext_ratio | - | 1.62 | - | - | 1.21 | - |
| ? | mixed_2dec_3ext_ar3b | - | 2862 | - | 3033 | 3094 | - |
| ? | mixed_2dec_3ext_dg | - | 2146 | - | 2827 | 3140 | - |
| ? | mixed_2dec_3ext_ratio | - | 1.33 | - | - | 0.99 | - |
| ? | uniform_extraction_ar3b | - | 813 | - | 913 | 944 | - |
| ? | uniform_extraction_dg | - | 417 | - | 775 | 946 | - |
| ? | uniform_extraction_ratio | - | 1.95 | - | - | 1.00 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256 | 100 | 808 | - | 1298 | 1298 | 1.61x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256 | 100 | 808 | - | 1298 | 1298 | 1.61x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256 | 100 | 419 | - | 1247 | 1247 | 2.98x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256 | 100 | 419 | - | 1247 | 1247 | 2.98x |

## DG (DiffusionGemma)

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| ? | max_new_tokens=256, comparator=comparator_ar3b_same_host | 500 | 807 | - | 1300 | 1625 | - |
| ? | max_new_tokens=256, denoising_steps=8 | 300 | 375 | - | 484 | - | - |
| ? | max_new_tokens=256, denoising_steps=32 | 300 | 416 | - | 829 | - | - |
| ? | max_new_tokens=256, denoising_steps=48_default | 2000 | 420 | - | 710 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=32 | 300 | 416 | - | 829 | 1396 | 3.36x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=8 | 300 | 375 | - | 484 | 5897 | 15.73x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256 | 2000 | 420 | - | 710 | 1282 | 3.17x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=1, latency_source=burst_wall_clock | 2000 | - | - | 472 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=2, latency_source=burst_wall_clock | 2000 | - | - | 529 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=4, latency_source=burst_wall_clock | 2000 | - | - | 1408 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=8, latency_source=burst_wall_clock | 2000 | - | - | 2123 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=16, latency_source=burst_wall_clock | 2000 | - | - | 3759 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=32, latency_source=burst_wall_clock | 2000 | - | - | 7770 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=64, latency_source=burst_wall_clock | 2000 | - | - | 15849 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=128, latency_source=burst_wall_clock | 2000 | - | - | 30878 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, batch_size=256, latency_source=burst_wall_clock | 2000 | - | - | 61494 | - | - |

## HDFS output-length sweep

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| LLaDA-8B-Instruct | direct | 50 | 2753 | - | 4465 | 4465 | 1.62x |
| LLaDA-8B-Instruct | hf_baseline | 50 | 1389 | - | 1556 | 1648 | - |
| LLaDA-8B-Instruct | direct | 50 | 10568 | - | 14334 | 14334 | 1.36x |
| LLaDA-8B-Instruct | hf_baseline | 50 | 1389 | - | 1556 | 1648 | - |
| LLaDA-8B-Instruct | direct | 50 | 11161 | - | 15023 | 15023 | 1.35x |
| LLaDA-8B-Instruct | hf_baseline | 50 | 1389 | - | 1556 | 1648 | - |
| LLaDA-8B-Instruct | direct | 50 | 10395 | - | 14617 | 14617 | 1.41x |
| LLaDA-8B-Instruct | hf_baseline | 50 | 1389 | - | 1556 | 1648 | - |
| LLaDA-8B-Instruct | direct | 50 | 4497 | - | 28452 | 28452 | 6.33x |
| LLaDA-8B-Instruct | hf_baseline | 50 | 1389 | - | 1556 | 1648 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=1024, prompt_token_count=947 | 100 | 6441 | - | 6478 | 6478 | 1.01x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, prompt_token_count=947 | 100 | 1617 | - | 1654 | 1654 | 1.02x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, prompt_token_count=947 | 100 | 3224 | - | 3263 | 3263 | 1.01x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, max_denoising_steps=8, prompt_token_count=1027 | 200 | 367 | - | 487 | 1216 | 3.32x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=2048, max_denoising_steps=8, prompt_token_count=1027 | 100 | 1192 | - | 2382 | 2382 | 2.00x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=8, prompt_token_count=1027 | 200 | 341 | - | 501 | 1202 | 3.52x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, max_denoising_steps=8, prompt_token_count=1027 | 200 | 469 | - | 808 | 1336 | 2.85x |

## Imputation

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| GSAI-ML/LLaDA-8B-Instruct | k_steps=8, k=1, field_f1_imputed=0.0, field_f1_all=0.0 | 500 | 823 | - | 987 | 1135 | - |
| GSAI-ML/LLaDA-8B-Instruct | k_steps=8, k=2, field_f1_imputed=0.413, field_f1_all=0.413 | 500 | 832 | - | 991 | 1074 | - |
| GSAI-ML/LLaDA-8B-Instruct | k_steps=8, k=4, field_f1_imputed=0.3011666666666667, field_f1_all=0.3011666666666667 | 500 | 824 | - | 990 | 1015 | - |
| GSAI-ML/LLaDA-8B-Instruct | k_steps=8, k=6, field_f1_imputed=0.15358634573487515, field_f1_all=0.15358634573487515 | 500 | 829 | - | 912 | 997 | - |
| GSAI-ML/LLaDA-8B-Instruct | k_steps=8, k=8, field_f1_imputed=0.22077633172650257, field_f1_all=0.22077633172650257 | 500 | 833 | - | 911 | 1004 | - |
| Qwen/Qwen2.5-3B-Instruct | k=1, field_f1_imputed=0.25, field_f1_all=0.25 | 500 | 917 | - | 1327 | 1629 | - |
| Qwen/Qwen2.5-3B-Instruct | k=2, field_f1_imputed=0.591, field_f1_all=0.591 | 500 | 878 | - | 1389 | 1627 | - |
| Qwen/Qwen2.5-3B-Instruct | k=4, field_f1_imputed=0.7434999999999998, field_f1_all=0.7434999999999998 | 500 | 903 | - | 1622 | 1629 | - |
| Qwen/Qwen2.5-3B-Instruct | k=6, field_f1_imputed=0.6430264476798156, field_f1_all=0.6430264476798156 | 500 | 828 | - | 1402 | 1618 | - |
| Qwen/Qwen2.5-3B-Instruct | k=8, field_f1_imputed=0.6882182589265036, field_f1_all=0.6882182589265036 | 500 | 929 | - | 1403 | 1627 | - |

## Infilling

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | suffix_identity_rate=0.0 | 50 | 922 | - | 1618 | - | - |
| Qwen/Qwen2.5-3B-Instruct | suffix_identity_rate=0.0 | 50 | 902 | - | 1133 | - | - |
| Qwen/Qwen2.5-3B-Instruct | suffix_identity_rate=0.0 | 50 | 901 | - | 1133 | - | - |
| Qwen/Qwen2.5-3B-Instruct | suffix_identity_rate=0.0 | 50 | 903 | - | 1199 | - | - |
| google/diffusiongemma-26B-A4B-it | suffix_identity_rate=0.0 | 50 | 334 | - | 1146 | - | - |
| google/diffusiongemma-26B-A4B-it | suffix_identity_rate=0.0 | 50 | 362 | - | 468 | - | - |
| google/diffusiongemma-26B-A4B-it | suffix_identity_rate=0.0 | 50 | 327 | - | 472 | - | - |
| google/diffusiongemma-26B-A4B-it | suffix_identity_rate=0.0 | 50 | 377 | - | 491 | - | - |

## Input-length sweep

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=0, prompt_token_count=992 | 100 | 1711 | - | 1720 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=2, prompt_token_count=1510 | 100 | 1731 | - | 1736 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=5, prompt_token_count=2287 | 100 | 1760 | - | 1764 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=10, prompt_token_count=3582 | 100 | 1816 | - | 1821 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=20, prompt_token_count=6172 | 100 | 1952 | - | 1959 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=0, prompt_token_count=992 | 100 | 1996 | - | 2182 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=2, prompt_token_count=1510 | 100 | 2010 | - | 2028 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=5, prompt_token_count=2287 | 100 | 2039 | - | 2048 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=10, prompt_token_count=3582 | 100 | 2098 | - | 2398 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=20, prompt_token_count=6172 | 100 | 2236 | - | 2249 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, schema_repeats=40, prompt_token_count=11352 | 100 | 20.8 | - | 25.3 | - | - |
| google/diffusiongemma-26B-A4B-it | denoising_steps=8, max_new_tokens=256, schema_repeats=0, prompt_token_count=1077 | 100 | 445 | - | 5991 | - | - |
| google/diffusiongemma-26B-A4B-it | denoising_steps=8, max_new_tokens=256, schema_repeats=2, prompt_token_count=1649 | 100 | 502 | - | 613 | - | - |
| google/diffusiongemma-26B-A4B-it | denoising_steps=8, max_new_tokens=256, schema_repeats=5, prompt_token_count=2507 | 100 | 599 | - | 715 | - | - |
| google/diffusiongemma-26B-A4B-it | denoising_steps=8, max_new_tokens=256, schema_repeats=10, prompt_token_count=3937 | 100 | 756 | - | 876 | - | - |
| google/diffusiongemma-26B-A4B-it | denoising_steps=8, max_new_tokens=256, schema_repeats=20, prompt_token_count=6797 | 100 | 1146 | - | 2043 | - | - |

## Iterative refinement

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | round=1 | 500 | 811 | - | 1300 | - | - |
| Qwen/Qwen2.5-3B-Instruct | round=2 | 500 | 808 | - | 1373 | - | - |
| Qwen/Qwen2.5-3B-Instruct | round=3 | 500 | 808 | - | 1372 | - | - |
| Qwen/Qwen2.5-3B-Instruct | round=1 | 200 | 1070 | - | 1231 | - | - |
| Qwen/Qwen2.5-3B-Instruct | round=2 | 200 | 829 | - | 849 | - | - |
| Qwen/Qwen2.5-3B-Instruct | round=3 | 200 | 828 | - | 848 | - | - |
| Qwen/Qwen2.5-7B-Instruct | round=1 | 200 | 1378 | - | 1823 | - | - |
| Qwen/Qwen2.5-7B-Instruct | round=2 | 200 | 1382 | - | 1795 | - | - |
| Qwen/Qwen2.5-7B-Instruct | round=3 | 200 | 1381 | - | 1794 | - | - |

## K-sweep (DG)

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=16, prompt_token_count=1077 | 200 | 382 | - | 716 | 1250 | 3.27x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=2, prompt_token_count=1077 | 200 | 202 | - | 301 | 1044 | 5.18x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=32, prompt_token_count=1077 | 200 | 422 | - | 1035 | 1387 | 3.29x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=4, prompt_token_count=1077 | 200 | 292 | - | 394 | 1116 | 3.82x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=64, prompt_token_count=1077 | 200 | 424 | - | 848 | 1298 | 3.06x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=8, prompt_token_count=1077 | 200 | 379 | - | 491 | 1211 | 3.20x |

## MDM (LLaDA)

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| GSAI-ML/LLaDA-8B-Instruct | k_steps=8 | 300 | 1102 | - | 1261 | - | - |

## MMPP

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=52, completed=52, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 52 | 726 | - | 1111 | 1111 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=243, completed=243, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 243 | 726 | - | 1276 | 1592 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 727 | - | 1271 | 1567 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 752 | - | 1322 | 1622 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=32.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 800 | - | 1343 | 1674 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=64.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 918 | - | 1508 | 1875 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=96.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 1307 | - | 2215 | 2708 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=128.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 2231 | - | 3487 | 4033 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=52, completed=52, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 52 | 726 | - | 1111 | 1111 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=243, completed=243, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 243 | 726 | - | 1276 | 1592 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 727 | - | 1271 | 1567 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 752 | - | 1322 | 1622 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=32.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 800 | - | 1343 | 1674 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=64.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 918 | - | 1508 | 1875 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=96.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 1307 | - | 2215 | 2708 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=128.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 2231 | - | 3487 | 4033 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=52, completed=52, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 52 | 22.6 | - | 60.8 | 60.8 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=243, completed=243, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 243 | 25.2 | - | 33.0 | 38.2 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 25.1 | - | 34.5 | 36.1 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 26.1 | - | 40.2 | 51.0 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=32.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 28.4 | - | 41.0 | 43.3 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=64.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 36.7 | - | 65.3 | 79.8 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=96.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 58.1 | - | 151 | 176 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=128.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 102 | - | 173 | 180 | - |

## Other

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| ? | k_steps=8 | 300 | - | - | - | - | - |
| ? | k_steps=8 | 300 | - | - | - | - | - |
| ? | k_steps=8 | 300 | 377 | - | 488 | - | - |
| LLaDA-8B-Instruct | direct | 300 | 10886 | - | 13717 | 14813 | 1.36x |
| LLaDA-8B-Instruct | hf_baseline | 300 | 1389 | - | 1556 | 1648 | - |
| Qwen/Qwen2.5-3B-Instruct | cells | 200 | 203 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | cells | 500 | 1077 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | measurements | 100 | 1617 | - | 1662 | - | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=1 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=2 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=4 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=8 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=16 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=24 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=32 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=48 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=64 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=96 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| Qwen2.5-1.5B-Instruct (K=8) | batch_size=128 | 200 | 23.3 | - | 27.3 | 27.3 | - |
| google/diffusiongemma-26B-A4B-it | measurements | 100 | 378 | - | 1229 | - | - |

## Output-length sweep

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=1024 | 300 | 6443 | - | 6455 | 6490 | 1.01x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=2048 | 300 | 12893 | - | 12916 | 12941 | 1.00x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128 | 300 | 813 | - | 821 | 859 | 1.06x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=1, latency_source=burst_wall_clock | 300 | - | - | 813 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=2, latency_source=burst_wall_clock | 300 | - | - | 795 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=4, latency_source=burst_wall_clock | 300 | - | - | 801 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=8, latency_source=burst_wall_clock | 300 | - | - | 820 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=16, latency_source=burst_wall_clock | 300 | - | - | 881 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=32, latency_source=burst_wall_clock | 300 | - | - | 984 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=64, latency_source=burst_wall_clock | 300 | - | - | 1205 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=128, latency_source=burst_wall_clock | 300 | - | - | 1679 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, batch_size=256, latency_source=burst_wall_clock | 300 | - | - | 2698 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512 | 300 | 3223 | - | 3234 | 3270 | 1.01x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=1, latency_source=burst_wall_clock | 300 | - | - | 3223 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=2, latency_source=burst_wall_clock | 300 | - | - | 3128 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=4, latency_source=burst_wall_clock | 300 | - | - | 3149 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=8, latency_source=burst_wall_clock | 300 | - | - | 3212 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=16, latency_source=burst_wall_clock | 300 | - | - | 3382 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=32, latency_source=burst_wall_clock | 300 | - | - | 3727 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=64, latency_source=burst_wall_clock | 300 | - | - | 4524 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=128, latency_source=burst_wall_clock | 300 | - | - | 6074 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, batch_size=256, latency_source=burst_wall_clock | 300 | - | - | 9594 | - | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, prompt_token_count=992 | 300 | 813 | - | 822 | 850 | 1.04x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, prompt_token_count=992 | 300 | 1616 | - | 1625 | 1653 | 1.02x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=32, prompt_token_count=992 | 300 | 210 | - | 219 | 247 | 1.18x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, prompt_token_count=992 | 300 | 3228 | - | 3237 | 3266 | 1.01x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128 | 300 | 813 | - | 821 | 855 | 1.05x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256 | 300 | 1618 | - | 1627 | 1664 | 1.03x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=32 | 300 | 211 | - | 218 | 254 | 1.21x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512 | 300 | 3226 | - | 3235 | 3274 | 1.01x |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64 | 300 | 411 | - | 419 | 457 | 1.11x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=1024 | 300 | 779 | - | 1113 | 2095 | 2.69x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=2048 | 300 | 1251 | - | 5361 | 5823 | 4.66x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128 | 300 | 420 | - | 770 | 1253 | 2.99x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, batch_size=1, latency_source=burst_wall_clock | 300 | - | - | 408 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, batch_size=2, latency_source=burst_wall_clock | 300 | - | - | 680 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, batch_size=4, latency_source=burst_wall_clock | 300 | - | - | 1241 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, batch_size=8, latency_source=burst_wall_clock | 300 | - | - | 1948 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512 | 300 | 524 | - | 965 | 1428 | 2.73x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, batch_size=1, latency_source=burst_wall_clock | 300 | - | - | 534 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, batch_size=2, latency_source=burst_wall_clock | 300 | - | - | 846 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, batch_size=4, latency_source=burst_wall_clock | 300 | - | - | 1747 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, batch_size=8, latency_source=burst_wall_clock | 300 | - | - | 2833 | - | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, max_denoising_steps=8, prompt_token_count=1077 | 300 | 379 | - | 490 | 1230 | 3.25x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=8, prompt_token_count=1077 | 300 | 379 | - | 492 | 1227 | 3.23x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=32, max_denoising_steps=8, prompt_token_count=1077 | 300 | 378 | - | 488 | 1241 | 3.28x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, max_denoising_steps=8, prompt_token_count=1077 | 300 | 501 | - | 827 | 1342 | 2.68x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128 | 300 | 418 | - | 769 | 1311 | 3.14x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256 | 300 | 419 | - | 769 | 1319 | 3.15x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=32 | 300 | 417 | - | 765 | 1389 | 3.33x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512 | 300 | 519 | - | 961 | 1412 | 2.72x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64 | 300 | 419 | - | 767 | 1403 | 3.35x |

## Output-length sweep (DG)

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| google/diffusiongemma-26B-A4B-it | max_new_tokens=1024, max_denoising_steps=8, prompt_token_count=1077 | 200 | 739 | - | 1119 | 1578 | 2.14x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=1280, max_denoising_steps=8, prompt_token_count=1077 | 200 | 860 | - | 1207 | 1697 | 1.97x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=1536, max_denoising_steps=8, prompt_token_count=1077 | 200 | 974 | - | 1811 | 2145 | 2.20x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=1792, max_denoising_steps=8, prompt_token_count=1077 | 200 | 1091 | - | 1478 | 1932 | 1.77x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=2048, max_denoising_steps=8, prompt_token_count=1077 | 200 | 1212 | - | 2886 | 3104 | 2.56x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=240, max_denoising_steps=8, prompt_token_count=1077 | 200 | 378 | - | 530 | 1238 | 3.27x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, max_denoising_steps=8, prompt_token_count=1077 | 200 | 378 | - | 1119 | 1215 | 3.21x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=264, max_denoising_steps=8, prompt_token_count=1077 | 200 | 500 | - | 828 | 1305 | 2.61x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=496, max_denoising_steps=8, prompt_token_count=1077 | 200 | 500 | - | 826 | 1330 | 2.66x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, max_denoising_steps=8, prompt_token_count=1077 | 200 | 501 | - | 1344 | 1471 | 2.94x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=528, max_denoising_steps=8, prompt_token_count=1077 | 200 | 620 | - | 1191 | 1444 | 2.33x |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=768, max_denoising_steps=8, prompt_token_count=1077 | 200 | 620 | - | 1188 | 1461 | 2.35x |

## Paper Table 3

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| AR-1.5B | steps=128 | - | 168 | - | 240 | 340 | - |
| AR-1.5B + grammar | steps=128 | - | 205 | - | 292 | 410 | - |
| AR-7B | steps=128 | - | 690 | - | 910 | 1240 | - |
| FlowFill-MDM (K=8) | steps=8 | - | 23.0 | - | 31.0 | 36.0 | - |
| Qwen2.5-1.5B-Instruct (K=8) | steps=128 | 500 | 23.1 | - | 27.3 | 29.1 | - |

## Perturbation robustness

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | abbreviation_i0.3 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | abbreviation_i1.0 | 200 | 808 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | case_variation_i0.3 | 200 | 812 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | case_variation_i1.0 | 200 | 813 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | clean | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | encoding_noise_i0.3 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | encoding_noise_i1.0 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | field_reorder_i0.3 | 200 | 806 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | field_reorder_i1.0 | 200 | 801 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | missing_field_i0.3 | 200 | 808 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | missing_field_i1.0 | 200 | 814 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | separator_change_i0.3 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | separator_change_i1.0 | 200 | 751 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | synonym_substitution_i0.3 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | synonym_substitution_i1.0 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | timestamp_format_i0.3 | 200 | 819 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | timestamp_format_i1.0 | 200 | 895 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | whitespace_i0.3 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | whitespace_i1.0 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | wrapping_noise_i0.3 | 200 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | wrapping_noise_i1.0 | 200 | 813 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | case_variation_i0.1 | 100 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | case_variation_i0.5 | 100 | 812 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | clean | 100 | 811 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | field_reorder_i0.1 | 100 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | field_reorder_i0.5 | 100 | 806 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | missing_field_i0.1 | 100 | 807 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | missing_field_i0.5 | 100 | 811 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | wrapping_noise_i0.1 | 100 | 811 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | wrapping_noise_i0.5 | 100 | 812 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | abbreviation_i1.0 | 100 | 1077 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | case_variation_i1.0 | 100 | 1078 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | clean | 100 | 1062 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | encoding_noise_i1.0 | 100 | 1077 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | field_reorder_i1.0 | 100 | 1076 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | missing_field_i1.0 | 100 | 1074 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | separator_change_i1.0 | 100 | 1079 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | synonym_substitution_i1.0 | 100 | 1077 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | timestamp_format_i1.0 | 100 | 1079 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | whitespace_i1.0 | 100 | 1076 | - | - | - | - |
| Qwen/Qwen2.5-3B-Instruct | wrapping_noise_i1.0 | 100 | 1078 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | case_variation_i0.1 | 100 | 1089 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | case_variation_i0.5 | 100 | 1093 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | clean | 100 | 1078 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | field_reorder_i0.1 | 100 | 1093 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | field_reorder_i0.5 | 100 | 1093 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | missing_field_i0.1 | 100 | 1092 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | missing_field_i0.5 | 100 | 1091 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | wrapping_noise_i0.1 | 100 | 1093 | - | - | - | - |
| Qwen/Qwen2.5-7B-Instruct | wrapping_noise_i0.5 | 100 | 1093 | - | - | - | - |

## Quality table

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| ? | steps=128 | 2000 | 168 | - | 240 | 340 | - |
| ? | steps=128 | 2000 | 205 | - | 292 | 410 | - |
| ? | steps=128 | 2000 | 690 | - | 910 | 1240 | - |
| ? | steps=8 | 2000 | 23.0 | - | 31.0 | 36.0 | - |
| ? | steps=128 | 2000 | 168 | - | 240 | 340 | - |
| ? | steps=128 | 2000 | 205 | - | 292 | 410 | - |
| ? | steps=128 | 2000 | 690 | - | 910 | 1240 | - |
| ? | steps=8 | 2000 | 23.0 | - | 31.0 | 36.0 | - |
| ? | steps=128 | 2000 | 168 | - | 240 | 340 | - |
| ? | steps=128 | 2000 | 205 | - | 292 | 410 | - |
| ? | steps=128 | 2000 | 690 | - | 910 | 1240 | - |
| ? | steps=8 | 2000 | 23.0 | - | 31.0 | 36.0 | - |
| ? | steps=128 | 200 | 168 | - | 240 | 340 | - |
| ? | steps=128 | 200 | 205 | - | 292 | 410 | - |
| ? | steps=128 | 200 | 690 | - | 910 | 1240 | - |
| ? | steps=8 | 200 | 23.0 | - | 31.0 | 36.0 | - |
| LLaDA-8B-Instruct | models | 2000 | 1389 | - | 1556 | 1648 | - |
| LLaDA-8B-Instruct | models | 200 | 1260 | - | 1495 | 1630 | - |
| Qwen2.5-1.5B-Instruct | models | 2000 | 1842 | - | 3053 | 4162 | - |
| Qwen2.5-1.5B-Instruct | models | 2000 | 1842 | - | 3053 | 4162 | - |
| Qwen2.5-7B-Instruct | models | 2000 | 2122 | - | 5480 | 7808 | - |
| Qwen2.5-7B-Instruct | models | 2000 | 2122 | - | 5480 | 7808 | - |

## Rate-lag curve

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=1.0, completion_ratio=1.0, sent=132, completed=132, errors=0, abandoned_at_drain=0 | 132 | 725 | - | 1293 | 1550 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=4.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 719 | - | 1267 | 1302 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=8.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 733 | - | 1284 | 1324 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=16.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 785 | - | 1364 | 1398 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=24.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 836 | - | 1430 | 1445 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=32.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 893 | - | 1525 | 1603 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=48.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 1231 | - | 1898 | 2365 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=64.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 1783 | - | 2980 | 3215 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=96.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 2828 | - | 4763 | 5330 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=128.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 3932 | - | 6617 | 6847 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=1.0, completion_ratio=1.0, sent=132, completed=132, errors=0, abandoned_at_drain=0 | 132 | 22.7 | - | 32.8 | 58.2 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=4.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 25.4 | - | 33.2 | 36.0 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=8.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 25.8 | - | 33.8 | 36.7 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=16.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 28.5 | - | 37.3 | 40.3 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=24.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 31.2 | - | 43.7 | 48.4 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=32.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 33.9 | - | 45.5 | 46.8 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=48.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 47.8 | - | 76.8 | 83.5 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=64.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 70.4 | - | 141 | 165 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=96.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 115 | - | 214 | 222 | - |
| Qwen/Qwen2.5-3B-Instruct | offered_rate_ev_s=128.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 122 | - | 1304 | 1346 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 814 | - | 858 | 858 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 804 | - | 822 | 823 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 801 | - | 816 | 820 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 813 | - | 849 | 853 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 866 | - | 885 | 888 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 23.6 | - | 58.8 | 58.8 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 25.3 | - | 31.1 | 32.2 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 25.8 | - | 30.1 | 31.7 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 26.3 | - | 31.6 | 33.0 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=128, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 28.0 | - | 35.5 | 36.3 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 1587 | - | 1663 | 1663 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 1585 | - | 1615 | 1620 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 1603 | - | 1675 | 1679 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 1698 | - | 1728 | 1731 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 1837 | - | 1936 | 1937 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 25.5 | - | 58.0 | 58.0 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 26.4 | - | 31.0 | 31.6 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 26.6 | - | 31.9 | 41.7 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 28.1 | - | 37.0 | 37.2 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 33.8 | - | 43.5 | 54.6 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 3161 | - | 3236 | 3236 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 3228 | - | 3340 | 3341 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 3364 | - | 3420 | 3421 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 3611 | - | 3790 | 3790 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 4457 | - | 4828 | 4835 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 27.0 | - | 58.5 | 58.5 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 27.9 | - | 34.4 | 35.1 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 28.7 | - | 34.5 | 35.5 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 32.7 | - | 44.1 | 54.5 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=512, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 45.1 | - | 64.2 | 67.5 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 417 | - | 454 | 454 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=2.0, completion_ratio=0.9917, sent=120, completed=119, errors=1, abandoned_at_drain=0 | 119 | 416 | - | 423 | 424 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 411 | - | 419 | 419 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 410 | - | 417 | 424 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 420 | - | 443 | 443 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 22.0 | - | 57.6 | 57.6 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=2.0, completion_ratio=0.9917, sent=120, completed=119, errors=1, abandoned_at_drain=0 | 119 | 22.7 | - | 31.0 | 32.0 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 24.1 | - | 28.7 | 29.4 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 24.6 | - | 33.5 | 40.4 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=64, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 26.4 | - | 32.4 | 34.3 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 949 | - | 1587 | 1776 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 949 | - | 1587 | 1776 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0, vip_n=0, bg_cancelled_by_vip=0 | 300 | 35.2 | - | 50.1 | 56.7 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 430 | - | 1202 | 1202 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 559 | - | 1801 | 1923 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 7947 | - | 13711 | 13874 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 16825 | - | 31838 | 32523 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=16.0, completion_ratio=0.8917, sent=120, completed=107, errors=0, abandoned_at_drain=13 | 107 | 18891 | - | 34796 | 35277 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 429 | - | 1201 | 1201 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 558 | - | 1801 | 1922 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 7947 | - | 13710 | 13873 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 16824 | - | 31837 | 32523 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=128, offered_rate_ev_s=16.0, completion_ratio=0.8917, sent=120, completed=107, errors=0, abandoned_at_drain=13 | 107 | 18890 | - | 34795 | 35276 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 398 | - | 1097 | 1097 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 569 | - | 2081 | 2294 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 7428 | - | 12560 | 12769 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 15639 | - | 29431 | 29590 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=0.9333, sent=120, completed=112, errors=0, abandoned_at_drain=8 | 112 | 17996 | - | 35005 | 35410 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 397 | - | 1096 | 1096 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 569 | - | 2081 | 2294 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 7427 | - | 12559 | 12768 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 15638 | - | 29430 | 29589 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=0.9333, sent=120, completed=112, errors=0, abandoned_at_drain=8 | 112 | 17995 | - | 35004 | 35409 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 599 | - | 2706 | 2706 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 1440 | - | 3992 | 4206 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 14375 | - | 27707 | 28076 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=8.0, completion_ratio=0.8167, sent=120, completed=98, errors=0, abandoned_at_drain=22 | 98 | 20154 | - | 36801 | 36801 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=16.0, completion_ratio=0.6417, sent=120, completed=77, errors=0, abandoned_at_drain=43 | 77 | 18247 | - | 38213 | 38213 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 500 | - | 2589 | 2589 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 1319 | - | 3872 | 4085 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 14255 | - | 27585 | 27954 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=8.0, completion_ratio=0.8167, sent=120, completed=98, errors=0, abandoned_at_drain=22 | 98 | 20031 | - | 36680 | 36680 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=512, offered_rate_ev_s=16.0, completion_ratio=0.6417, sent=120, completed=77, errors=0, abandoned_at_drain=43 | 77 | 18123 | - | 38090 | 38090 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 432 | - | 987 | 987 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 520 | - | 1549 | 1885 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 7492 | - | 12448 | 12566 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 17614 | - | 31620 | 31865 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=16.0, completion_ratio=0.9333, sent=120, completed=112, errors=0, abandoned_at_drain=8 | 112 | 18048 | - | 35016 | 35335 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=1.0, completion_ratio=1.0, sent=81, completed=81, errors=0, abandoned_at_drain=0 | 81 | 432 | - | 986 | 986 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=2.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 519 | - | 1548 | 1884 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=4.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 7492 | - | 12447 | 12565 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=8.0, completion_ratio=1.0, sent=120, completed=120, errors=0, abandoned_at_drain=0 | 120 | 17613 | - | 31619 | 31864 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=64, offered_rate_ev_s=16.0, completion_ratio=0.9333, sent=120, completed=112, errors=0, abandoned_at_drain=8 | 112 | 18047 | - | 35015 | 35334 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=0.5, completion_ratio=1.0, sent=65, completed=65, errors=0, abandoned_at_drain=0 | 65 | 402 | - | 5605 | 5605 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=1.0, completion_ratio=1.0, sent=132, completed=132, errors=0, abandoned_at_drain=0 | 132 | 428 | - | 1011 | 1040 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=2.0, completion_ratio=0.2014, sent=293, completed=59, errors=234, abandoned_at_drain=0 | 59 | 449 | - | 1162 | 1162 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=3.0, completion_ratio=0.0, sent=300, completed=0, errors=300, abandoned_at_drain=0 | 0 | 0.00 | - | 0.00 | 0.00 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=4.0, completion_ratio=0.0, sent=300, completed=0, errors=300, abandoned_at_drain=0 | 0 | 0.00 | - | 0.00 | 0.00 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=0.5, completion_ratio=1.0, sent=65, completed=65, errors=0, abandoned_at_drain=0 | 65 | 402 | - | 5604 | 5604 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=1.0, completion_ratio=1.0, sent=132, completed=132, errors=0, abandoned_at_drain=0 | 132 | 427 | - | 1011 | 1040 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=2.0, completion_ratio=0.2014, sent=293, completed=59, errors=234, abandoned_at_drain=0 | 59 | 479 | - | 1161 | 1161 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=3.0, completion_ratio=0.0, sent=300, completed=0, errors=300, abandoned_at_drain=0 | 0 | 0.00 | - | 0.00 | 0.00 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=4.0, completion_ratio=0.0, sent=300, completed=0, errors=300, abandoned_at_drain=0 | 0 | 0.00 | - | 0.00 | 0.00 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=0.5, completion_ratio=1.0, sent=65, completed=65, errors=0, abandoned_at_drain=0 | 65 | 389 | - | 740 | 740 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=1.0, completion_ratio=1.0, sent=132, completed=132, errors=0, abandoned_at_drain=0 | 132 | 434 | - | 1167 | 1246 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=1.5, completion_ratio=0.9954, sent=216, completed=215, errors=1, abandoned_at_drain=0 | 215 | 566 | - | 2703 | 3007 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=2.0, completion_ratio=1.0, sent=293, completed=293, errors=0, abandoned_at_drain=0 | 293 | 810 | - | 3613 | 3963 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=2.5, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 4633 | - | 10574 | 10706 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=3.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 8575 | - | 24867 | 25068 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=4.0, completion_ratio=0.9733, sent=300, completed=292, errors=0, abandoned_at_drain=8 | 292 | 21320 | - | 46612 | 47279 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=6.0, completion_ratio=0.7833, sent=300, completed=235, errors=0, abandoned_at_drain=65 | 235 | 24162 | - | 54380 | 54782 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=0.5, completion_ratio=1.0, sent=65, completed=65, errors=0, abandoned_at_drain=0 | 65 | 389 | - | 740 | 740 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=1.0, completion_ratio=1.0, sent=132, completed=132, errors=0, abandoned_at_drain=0 | 132 | 433 | - | 1167 | 1246 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=1.5, completion_ratio=0.9954, sent=216, completed=215, errors=1, abandoned_at_drain=0 | 215 | 565 | - | 2703 | 3006 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=2.0, completion_ratio=1.0, sent=293, completed=293, errors=0, abandoned_at_drain=0 | 293 | 809 | - | 3613 | 3962 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=2.5, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 4632 | - | 10574 | 10706 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=3.0, completion_ratio=1.0, sent=300, completed=300, errors=0, abandoned_at_drain=0 | 300 | 8574 | - | 24867 | 25067 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=4.0, completion_ratio=0.9733, sent=300, completed=292, errors=0, abandoned_at_drain=8 | 292 | 21319 | - | 46612 | 47278 | - |
| google/diffusiongemma-26B-A4B-it | offered_rate_ev_s=6.0, completion_ratio=0.7833, sent=300, completed=235, errors=0, abandoned_at_drain=65 | 235 | 24161 | - | 54379 | 54781 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0494, sent=81, completed=85, errors=1, abandoned_at_drain=0, vip_n=4, bg_cancelled_by_vip=1 | 85 | 397 | - | 1053 | 1053 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0494, sent=81, completed=85, errors=1, abandoned_at_drain=0, vip_n=4, bg_cancelled_by_vip=1 | 85 | 397 | - | 1053 | 1053 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0494, sent=81, completed=85, errors=1, abandoned_at_drain=0, vip_n=4, bg_cancelled_by_vip=1 | 85 | 397 | - | 1052 | 1052 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0494, sent=81, completed=85, errors=1, abandoned_at_drain=0, vip_n=4, bg_cancelled_by_vip=1 | 85 | 439 | - | 872 | 872 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0494, sent=81, completed=85, errors=0, abandoned_at_drain=0, vip_n=4, bg_cancelled_by_vip=0 | 85 | 431 | - | 1381 | 1381 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0494, sent=81, completed=85, errors=0, abandoned_at_drain=0, vip_n=4, bg_cancelled_by_vip=0 | 85 | 405 | - | 1381 | 1381 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0494, sent=81, completed=85, errors=0, abandoned_at_drain=0, vip_n=4, bg_cancelled_by_vip=0 | 85 | 405 | - | 1381 | 1381 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0494, sent=81, completed=85, errors=0, abandoned_at_drain=0, vip_n=4, bg_cancelled_by_vip=0 | 85 | 395 | - | 730 | 730 | - |

## VIP probing

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0044, sent=1591, completed=1598, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 1598 | 735 | - | 1278 | 1619 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0022, sent=3157, completed=3164, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 3164 | 787 | - | 1372 | 1735 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=32.0, completion_ratio=1.0009, sent=6346, completed=6352, errors=1, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 6352 | 934 | - | 1611 | 2051 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=48.0, completion_ratio=1.0001, sent=9634, completed=9635, errors=6, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 9635 | 1540 | - | 15292 | 22853 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0044, sent=1591, completed=1598, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 1598 | 735 | - | 1278 | 1619 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0022, sent=3157, completed=3164, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 3164 | 787 | - | 1372 | 1735 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=32.0, completion_ratio=1.0009, sent=6346, completed=6352, errors=1, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 6352 | 934 | - | 1611 | 2051 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=48.0, completion_ratio=1.0001, sent=9634, completed=9635, errors=6, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 9635 | 1540 | - | 15292 | 22853 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0044, sent=1591, completed=1598, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 1598 | 25.8 | - | 33.4 | 39.5 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0022, sent=3157, completed=3164, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 3164 | 28.3 | - | 37.2 | 44.3 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=32.0, completion_ratio=1.0009, sent=6346, completed=6352, errors=1, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 6352 | 35.4 | - | 50.0 | 58.9 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=48.0, completion_ratio=1.0001, sent=9634, completed=9635, errors=6, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 9635 | 62.4 | - | 11174 | 15941 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=8.0, completion_ratio=1.0044, sent=1591, completed=1598, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 1598 | 762 | - | 845 | 845 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=16.0, completion_ratio=1.0022, sent=3157, completed=3164, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 3164 | 803 | - | 934 | 934 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=32.0, completion_ratio=1.0009, sent=6346, completed=6352, errors=1, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 6352 | 1013 | - | 1134 | 1134 | - |
| Qwen/Qwen2.5-3B-Instruct | max_new_tokens=256, offered_rate_ev_s=48.0, completion_ratio=1.0001, sent=9634, completed=9635, errors=6, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 9635 | 1710 | - | 10344 | 10344 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=0.5, completion_ratio=1.0795, sent=88, completed=95, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 95 | 391 | - | 1147 | 1147 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0366, sent=191, completed=198, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 198 | 425 | - | 1401 | 1506 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.5, completion_ratio=1.0239, sent=293, completed=300, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 300 | 520 | - | 3437 | 3774 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=0.5, completion_ratio=1.0795, sent=88, completed=95, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 95 | 391 | - | 1147 | 1147 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0366, sent=191, completed=198, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 198 | 425 | - | 1401 | 1506 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.5, completion_ratio=1.0239, sent=293, completed=300, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 300 | 520 | - | 3437 | 3774 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=0.5, completion_ratio=1.0795, sent=88, completed=95, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 95 | 391 | - | 1147 | 1147 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0366, sent=191, completed=198, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 198 | 425 | - | 1400 | 1506 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.5, completion_ratio=1.0239, sent=293, completed=300, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 300 | 520 | - | 3436 | 3773 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=0.5, completion_ratio=1.0795, sent=88, completed=95, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 95 | 405 | - | 483 | 483 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.0, completion_ratio=1.0366, sent=191, completed=198, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 198 | 396 | - | 488 | 488 | - |
| google/diffusiongemma-26B-A4B-it | max_new_tokens=256, offered_rate_ev_s=1.5, completion_ratio=1.0239, sent=293, completed=300, errors=0, abandoned_at_drain=0, vip_n=7, bg_cancelled_by_vip=0 | 300 | 432 | - | 770 | 770 | - |

## Wide-schema sweep

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-3B-Instruct | n_fields=12, level=12 | 200 | 738 | - | 1280 | 1523 | - |
| Qwen/Qwen2.5-3B-Instruct | n_fields=25, level=25 | 200 | 1440 | - | 1837 | 1849 | - |
| Qwen/Qwen2.5-3B-Instruct | n_fields=50, level=50 | 200 | 3122 | - | 3411 | 3494 | - |
| Qwen/Qwen2.5-3B-Instruct | n_fields=100, level=100 | 200 | 6239 | - | 7312 | 8457 | - |
| google/diffusiongemma-26B-A4B-it | n_fields=12, level=12 | 200 | 417 | - | 1251 | 1372 | - |
| google/diffusiongemma-26B-A4B-it | n_fields=25, level=25 | 200 | 589 | - | 1809 | 1865 | - |
| google/diffusiongemma-26B-A4B-it | n_fields=50, level=50 | 200 | 1037 | - | 2224 | 2298 | - |
| google/diffusiongemma-26B-A4B-it | n_fields=100, level=100 | 200 | 2142 | - | 3318 | 3381 | - |

## vLLM diagnostics

| Model | Condition | n | p50 | p95 | p99 | p99.9 | max/med |
|-------|-----------|---|-----|-----|-----|-------|---------|
| Qwen/Qwen2.5-1.5B-Instruct | max_new_tokens=256 | 100 | 419 | - | 571 | 571 | 1.36x |

*\* = computed from raw_latencies_ms array*
