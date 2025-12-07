sudo -u azureuser python3 - <<'PY'
import json
from pathlib import Path
path = Path('/home/azureuser/qwen/results/qwen2_5_32b_fp16_vllm_20k.json')
data = json.loads(path.read_text())
for key in [
    'model',
    'precision',
    'prompt_tokens',
    'generated_tokens',
    'wall_time_seconds',
    'tokens_per_second',
    'engine',
    'tensor_parallel_size',
    'max_model_len',
    'gpu_memory_utilization'
]:
    print(f"{key}: {data.get(key)}")
PY
