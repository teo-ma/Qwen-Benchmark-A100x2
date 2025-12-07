sudo -u azureuser python3 - <<'PY'
from pathlib import Path
text = Path('/home/azureuser/qwen/qwen_multi_benchmark.py').read_text().splitlines()
for idx, line in enumerate(text, 1):
    if 330 <= idx <= 390:
        print(f"{idx:04d}: {line}")
PY
