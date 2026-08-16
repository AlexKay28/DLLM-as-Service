#!/bin/bash
# Activate the GPU stack for this project
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/home/alexkay28/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/alexkay28/projects/DLLM-as-Service:$PYTHONPATH"
