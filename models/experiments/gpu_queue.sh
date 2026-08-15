#!/usr/bin/env bash
# GPU experiment queue for #57, #44, #68
# Run after the AR anchors complete.
# All experiments use .venv-dg (vLLM 0.25.1, pinned stack).

set -e
cd /home/alexkay28/projects/DLLM-as-Service

PY=.venv-dg/bin/python
export HF_HUB_OFFLINE=1

echo "============================================"
echo "GPU Queue: #57 DG anchors + #44 + #68"
echo "Started: $(date)"
echo "============================================"

# --- #57: DG anchor re-measurement ---
echo ""
echo "=== #57: DG anchors (5×2 cells, K=8, n=300) ==="
$PY models/experiments/run_anchor_remeasure.py --arm dg --repeats 5 --n 300 --lengths 256 512 --k-steps 8
echo "DG anchors done: $(date)"

# --- #44 Arm A: input-length sweep AR-3B ---
echo ""
echo "=== #44: AR-3B input-length sweep ==="
$PY models/experiments/run_input_length_sweep.py \
    --model Qwen/Qwen2.5-3B-Instruct --n 100 \
    --output results/issue44_ar3b_input_sweep.json
echo "AR input sweep done: $(date)"

# --- #44 Arm A: input-length sweep DG ---
echo ""
echo "=== #44: DG input-length sweep (K=8) ==="
$PY models/experiments/run_input_length_sweep.py \
    --model google/diffusiongemma-26B-A4B-it --denoising-steps 8 --n 100 \
    --output results/issue44_dg_input_sweep.json
echo "DG input sweep done: $(date)"

# --- #44 Arm B: DG K-sweep (base decomposition) ---
echo ""
echo "=== #44: DG K-sweep (base decomposition) ==="
for K in 2 4 8 16 32 64; do
    echo "  K=$K..."
    $PY models/experiments/run_vllm_serving.py \
        --model google/diffusiongemma-26B-A4B-it \
        --denoising-steps $K --n 200 --skip-throughput \
        --output results/issue44_dg_ksweep_K${K}.json \
        --save-predictions results/issue44_dg_ksweep_K${K}_preds.jsonl
done
echo "K-sweep done: $(date)"

# --- #68: Dense canvas boundary sweep ---
echo ""
echo "=== #68: Canvas boundary sweep (DG, K=8) ==="
# Run at lengths straddling canvas boundaries
for L in 240 256 264 496 512 528 768 1024 1280 1536 1792 2048; do
    echo "  L=$L..."
    $PY models/experiments/run_vllm_serving.py \
        --model google/diffusiongemma-26B-A4B-it \
        --denoising-steps 8 --max-new-tokens $L --min-tokens $L \
        --n 200 --skip-throughput \
        --output results/issue68_dg_L${L}.json \
        --save-predictions results/issue68_dg_L${L}_preds.jsonl
done
echo "Boundary sweep done: $(date)"

echo ""
echo "============================================"
echo "GPU Queue complete: $(date)"
echo "============================================"
