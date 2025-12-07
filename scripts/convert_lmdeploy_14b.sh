#!/bin/bash
set -euo pipefail

sudo -u azureuser /home/azureuser/qwen-env/bin/python - <<'PY'
from lmdeploy.turbomind.deploy.converter import main

main(
    model_name="qwen",
    model_path="Qwen/Qwen2.5-14B-Instruct",
    tp=2,
    dst_path="/home/azureuser/qwen/lmdeploy/qwen2_5_14b_tp2",
    trust_remote_code=True,
)
PY
