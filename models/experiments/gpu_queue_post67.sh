#!/usr/bin/env bash
# GPU queue: #5 energy → #6 memory → #7 infilling
# Runs after #67 LLaDA completes.
set -e
cd /home/alexkay28/projects/DLLM-as-Service
export HF_HUB_OFFLINE=1
PY=.venv-dg/bin/python

echo "=== Post-#67 GPU queue started at $(date) ==="

# --- #5: Energy measurement ---
echo "=== #5: Energy at compute-matched pair ==="
$PY models/experiments/run_energy_measurement.py --n 100 --L 256

# --- #6: Long-context memory probe ---
echo "=== #6: #58 P5 memory probe ==="
$PY models/experiments/run_memory_probe.py

# --- #7: Infilling experiment ---
echo "=== #7: #59 Stage 1 infilling ==="
$PY models/experiments/run_infilling.py

echo "=== Post-#67 queue complete at $(date) ==="
