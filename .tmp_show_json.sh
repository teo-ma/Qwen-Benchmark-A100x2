sudo -u azureuser python3 - <<'PY'
import json
from pathlib import Path
path = Path('/home/azureuser/qwen/results/qwen2_5_32b_fp16_vllm_20k.json')
data = json.loads(path.read_text())
print(json.dumps(data, indent=2))
PY
