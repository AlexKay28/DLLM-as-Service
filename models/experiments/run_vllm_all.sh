#!/usr/bin/env bash
# Issue #12: full vLLM serving sweep for the AR baselines.
#
# One process per model: vLLM's engine runs in a subprocess and releasing an
# 80GB-resident 7B cleanly mid-script is more trouble than just exiting.
#
# n=2000 matches the existing HF-transformers quality runs
# (results/quality_table2_ar_n2000.json) so Field-F1 is directly comparable and
# any latency difference is attributable to serving, not to the sample.
set -u

cd "$(dirname "$0")/../.."
PY=.venv-dllm/bin/python
export HF_HUB_OFFLINE=1

BATCHES="1 2 4 8 16 32 64 128 256 512"

for m in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  tag=$(echo "$m" | sed 's#.*/##; s/Qwen2.5-//; s/-Instruct//' | tr 'A-Z' 'a-z')
  echo "===== $m -> results/vllm_serving_${tag}.json ====="
  $PY models/experiments/run_vllm_serving.py \
      --model "$m" --n 2000 \
      --batch-sizes $BATCHES --throughput-repeats 3 \
      --output "results/vllm_serving_${tag}.json" \
    || echo "FAILED: $m"
done

# Prefix caching is the paper's frozen-T conditioning cache, credited to the AR
# baselines. This run removes it to price that credit.
echo "===== 1.5B, prefix caching OFF ====="
$PY models/experiments/run_vllm_serving.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --n 500 --no-prefix-caching \
    --batch-sizes $BATCHES --throughput-repeats 3 \
    --output results/vllm_serving_1.5b_noprefix.json \
  || echo "FAILED: no-prefix run"

echo "===== done ====="
