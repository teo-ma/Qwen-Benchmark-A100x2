#!/bin/bash
set -euo pipefail
sudo -u azureuser bash -lc '
set -euo pipefail
source ~/qwen-env/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p ~/qwen/results
cd ~/qwen
MASTER_PORT="29535"
torchrun --nproc_per_node=2 --master_port "$MASTER_PORT" \
  qwen_multi_benchmark.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --precision FP16 \
  --prompt-tokens 10000 \
  --generated-tokens 800 \
  --engine deepspeed \
  --tensor-parallel-size 2 \
  --output-json ~/qwen/results/qwen2_5_32b_fp16_deepspeed_10k.json
'
