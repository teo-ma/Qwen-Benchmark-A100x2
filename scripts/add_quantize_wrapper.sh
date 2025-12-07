#!/bin/bash
set -euo pipefail

cat <<'PY' >/home/azureuser/TensorRT-LLM/examples/quantization/quantize_with_patch.py
#!/usr/bin/env python
import runpy
import sys

try:
    import transformers  # type: ignore
    if not hasattr(transformers.modeling_utils, 'Conv1D'):
        from transformers.models.gpt2.modeling_gpt2 import Conv1D as _LegacyConv1D
        transformers.modeling_utils.Conv1D = _LegacyConv1D  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover
    print(f"[quantize_with_patch] Warning: failed to set Conv1D attr: {exc}", file=sys.stderr)

runpy.run_path('/home/azureuser/TensorRT-LLM/examples/quantization/quantize.py', run_name='__main__')
PY

chmod +x /home/azureuser/TensorRT-LLM/examples/quantization/quantize_with_patch.py
