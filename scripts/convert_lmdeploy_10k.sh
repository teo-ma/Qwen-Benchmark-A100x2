#!/bin/bash
set -euo pipefail
sudo -u azureuser bash -lc '
set -euo pipefail
source ~/qwen-env/bin/activate
rm -rf ~/qwen/lmdeploy/qwen2_5_32b_tp2_ctx14k
lmdeploy convert turbomind Qwen/Qwen2.5-32B-Instruct \
  --tp 2 \
  --session-len 14336 \
  --dtype fp16 \
  --work-dir ~/qwen/lmdeploy/qwen2_5_32b_tp2_ctx14k
'
