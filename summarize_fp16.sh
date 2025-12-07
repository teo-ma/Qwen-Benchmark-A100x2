sudo -u azureuser bash -lc '
python3 - <<'PY'
import json
from pathlib import Path
path = Path("/home/azureuser/qwen/results/qwen2_5_72b_fp16_tp2_flash.json")
data = json.loads(path.read_text())
subset = {k: data[k] for k in ("wall_time_seconds", "tokens_per_second", "generated_tokens", "prompt_tokens", "peak_memory_gib")}
print(json.dumps(subset, ensure_ascii=False, indent=2))
PY
'
