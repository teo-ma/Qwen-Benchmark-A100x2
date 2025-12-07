#!/bin/bash
set -euo pipefail

# Allow overriding the Python env used for vLLM via $VLLM_ENV_PATH.
VLLM_ENV_PATH="${VLLM_ENV_PATH:-/home/azureuser/vllm-env}"

sudo -u azureuser bash -lc "
set -euo pipefail
if [ ! -d '${VLLM_ENV_PATH}' ]; then
  echo 'Expected vLLM virtualenv at ${VLLM_ENV_PATH} but it was not found.' >&2
  exit 1
fi
source '${VLLM_ENV_PATH}'/bin/activate
mkdir -p ~/qwen/results
python ~/qwen/qwen_multi_benchmark.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --precision FP16 \
  --prompt-tokens 10000 \
  --generated-tokens 800 \
  --engine vllm \
  --tensor-parallel-size 2 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.9 \
  --output-json ~/qwen/results/qwen2_5_14b_fp16_vllm_10k.json
"
