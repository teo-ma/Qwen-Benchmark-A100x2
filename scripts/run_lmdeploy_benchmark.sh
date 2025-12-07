sudo -u azureuser bash -lc '
set -euo pipefail
source ~/qwen-env/bin/activate
mkdir -p ~/qwen/results
python ~/qwen/qwen_multi_benchmark.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --precision FP16 \
  --prompt-tokens 20000 \
  --generated-tokens 1000 \
  --engine lmdeploy \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --lmdeploy-model-path ~/qwen/lmdeploy/qwen2_5_32b_tp2 \
  --output-json ~/qwen/results/qwen2_5_32b_fp16_lmdeploy_20k.json
'
