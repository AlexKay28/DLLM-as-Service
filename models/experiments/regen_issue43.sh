#!/bin/bash
# Issue #43 — regenerate L-sweep with saved predictions for parser re-scoring.
#
# Two arms:
#   Arm 1: forced min_tokens=L (reproduces published config, now with preds saved)
#   Arm 2: natural stopping (min_tokens=0, max_tokens=L) — deployment-realistic
#
# Then re-score all cells under three parser variants:
#   python models/experiments/rescore_parsers.py results/p2_{model}_L{L}[_natural]_preds.jsonl
#
# Usage:  bash models/experiments/regen_issue43.sh
# Requires: GPU + vLLM 0.25.1 + .venv-dg

set -euo pipefail
PYTHON="${PYTHON:-.venv-dg/bin/python}"
PYTHON="HF_HUB_OFFLINE=1 $PYTHON"
MODELS=(google/diffusiongemma-26B-A4B-it Qwen/Qwen2.5-3B-Instruct)
LEVELS=(128 256 512 1024 2048)
N=300

for L in "${LEVELS[@]}"; do
  for model in "${MODELS[@]}"; do
    tag=$(echo "$model" | sed 's|.*/||; s|-Instruct||; s|-A4B-it||; s|Qwen2.5|ar3b|; s|diffusiongemma|dg|')

    echo "=== L=$L  model=$tag  forced min_tokens=$L ==="
    eval "$PYTHON models/experiments/run_vllm_serving.py \
      --model $model --n $N --min-tokens $L --max-new-tokens $L \
      --save-predictions results/p2_${tag}_L${L}_preds.jsonl \
      --output results/p2_${tag}_L${L}.json --skip-throughput"

    echo "=== L=$L  model=$tag  natural stopping (min_tokens=0) ==="
    eval "$PYTHON models/experiments/run_vllm_serving.py \
      --model $model --n $N --min-tokens 0 --max-new-tokens $L \
      --save-predictions results/p2_${tag}_L${L}_natural_preds.jsonl \
      --output results/p2_${tag}_L${L}_natural.json --skip-throughput"
  done
done

echo ""
echo "=== Re-scoring all cells under three parser variants ==="
for L in "${LEVELS[@]}"; do
  for tag in dg ar3b; do
    for suffix in "" "_natural"; do
      preds="results/p2_${tag}_L${L}${suffix}_preds.jsonl"
      if [ -f "$preds" ]; then
        echo ""
        echo "--- L=$L  $tag  suffix=$suffix ---"
        python3 models/experiments/rescore_parsers.py "$preds"
      fi
    done
  done
done
