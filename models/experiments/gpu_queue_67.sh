#!/usr/bin/env bash
# #67: Generalization — second corpus (HDFS) + third model (LLaDA)
set -e
cd /home/alexkay28/projects/DLLM-as-Service
export HF_HUB_OFFLINE=1

DG=.venv-dg/bin/python
DINFER=.venv-dinfer/bin/python

echo "=== #67 Generalization started at $(date) ==="

# --- Arm (a): DG on HDFS ---
echo "=== Arm (a): DiffusionGemma on HDFS ==="
for L in 128 256 512 1024 2048; do
    echo "  DG HDFS L=$L..."
    $DG models/experiments/run_vllm_serving.py \
        --model google/diffusiongemma-26B-A4B-it \
        --denoising-steps 8 --source hdfs \
        --max-new-tokens $L --min-tokens $L \
        --n 200 --skip-throughput \
        --output results/issue67_dg_hdfs_L${L}.json \
        --save-predictions results/issue67_dg_hdfs_L${L}_preds.jsonl
done

# Also re-run AR-3B on HDFS on pinned stack for comparison
echo "=== AR-3B on HDFS (pinned stack) ==="
for L in 128 256 512 1024 2048; do
    echo "  AR HDFS L=$L..."
    $DG models/experiments/run_vllm_serving.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --source hdfs --max-new-tokens $L --min-tokens $L \
        --n 200 --skip-throughput \
        --output results/issue67_ar3b_hdfs_L${L}.json \
        --save-predictions results/issue67_ar3b_hdfs_L${L}_preds.jsonl
done

# --- Arm (c): LLaDA-8B under dInfer, block_length sweep ---
echo "=== Arm (c): LLaDA-8B under dInfer ==="
# Sweep gen_len at block_length=32
for GEN in 128 256 512 1024; do
    echo "  LLaDA gen_len=$GEN block_length=32..."
    $DINFER models/experiments/run_dinfer_serving.py \
        --n 100 --gen-len $GEN --block-length 32 --cache '' \
        --source bgl \
        --out issue67_llada_gen${GEN}_bl32.json
done

# Sweep block_length at gen_len=256
for BL in 32 64 128 256; do
    echo "  LLaDA gen_len=256 block_length=$BL..."
    $DINFER models/experiments/run_dinfer_serving.py \
        --n 100 --gen-len 256 --block-length $BL --cache '' \
        --source bgl \
        --out issue67_llada_gen256_bl${BL}.json
done

echo "=== #67 complete at $(date) ==="
