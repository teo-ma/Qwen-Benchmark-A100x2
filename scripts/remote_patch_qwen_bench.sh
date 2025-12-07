#!/bin/bash
set -euo pipefail

cat <<'PY' >/tmp/patch_qwen_bench.py
from pathlib import Path

path = Path('/home/azureuser/qwen/qwen_multi_benchmark.py')
text = path.read_text()
changed = False

if 'lmdeploy_pipeline' not in text:
    needle = "    LLM = None\n    SamplingParams = None\n\nBASE_PROMPT = ("
    replacement = "    LLM = None\n    SamplingParams = None\n\ntry:  # LMDeploy is optional unless TurboMind benchmarking is requested\n    from lmdeploy import pipeline as lmdeploy_pipeline  # type: ignore\n    from lmdeploy import TurbomindEngineConfig  # type: ignore\nexcept ImportError:  # pragma: no cover - LMDeploy only needed for TurboMind runs\n    lmdeploy_pipeline = None\n    TurbomindEngineConfig = None\n\nBASE_PROMPT = ("
    if needle not in text:
        raise SystemExit('failed to locate vLLM import block for LMDeploy insertion')
    text = text.replace(needle, replacement, 1)
    changed = True

if 'lmdeploy_model_path: Optional[str]' not in text:
    sig_old = "    gpu_memory_utilization: float,\n) -> Dict[str, object]:"
    sig_new = "    gpu_memory_utilization: float,\n    lmdeploy_model_path: Optional[str],\n) -> Dict[str, object]:"
    if sig_old not in text:
        raise SystemExit('failed to update run_benchmark signature')
    text = text.replace(sig_old, sig_new, 1)
    changed = True

if 'engine == "lmdeploy"' not in text:
    marker = "        peak_memory = collect_peak_memory()\n        max_memory_request = None\n\n    else:"
    block = "        peak_memory = collect_peak_memory()\n        max_memory_request = None\n\n    elif engine == \"lmdeploy\":\n        if precision.lower() != \"fp16\":\n            raise ValueError(\"LMDeploy TurboMind path currently supports FP16 only in this script\")\n        if lmdeploy_pipeline is None or TurbomindEngineConfig is None:\n            raise RuntimeError(\"lmdeploy package is required when --engine lmdeploy\")\n        if not lmdeploy_model_path:\n            raise ValueError(\"--lmdeploy-model-path is required when --engine lmdeploy\")\n\n        session_len = max_model_len or (prompt_tokens + gen_tokens + 1024)\n        cache_ratio = max(0.1, min(gpu_memory_utilization, 0.95))\n        backend_config = TurbomindEngineConfig(\n            tp=tensor_parallel_size,\n            session_len=session_len,\n            cache_max_entry_count=cache_ratio,\n            max_batch_size=1,\n        )\n\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n            torch.cuda.reset_peak_memory_stats()\n\n        pipe = lmdeploy_pipeline(\n            lmdeploy_model_path,\n            backend_config=backend_config,\n        )\n\n        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)\n\n        sync_cuda()\n        start = time.perf_counter()\n        responses = pipe(\n            [prompt_text],\n            max_new_tokens=gen_tokens,\n            temperature=0.7,\n            top_p=0.9,\n        )\n        sync_cuda()\n        elapsed = time.perf_counter() - start\n\n        if not responses:\n            raise RuntimeError(\"LMDeploy pipeline returned no responses\")\n        first_response = responses[0]\n        new_tokens = getattr(first_response, \"generate_token_len\", 0)\n        if not new_tokens and getattr(first_response, \"token_ids\", None):\n            new_tokens = len(first_response.token_ids)\n        if not new_tokens:\n            raise RuntimeError(\"LMDeploy response missing token statistics\")\n        excerpt_text = str(getattr(first_response, \"text\", \"\")).strip()\n        device_map_serializable = {\n            \"engine\": \"lmdeploy-turbomind\",\n            \"tensor_parallel_size\": tensor_parallel_size,\n            \"session_len\": session_len,\n        }\n        peak_memory = collect_peak_memory()\n        max_memory_request = None\n\n    else:"
    if marker not in text:
        raise SystemExit('failed to locate vLLM tail marker for LMDeploy block')
    text = text.replace(marker, block, 1)
    changed = True

if '"lmdeploy_model_path"' not in text:
    result_needle = '        "gpu_memory_utilization": gpu_memory_utilization,\n    }'
    result_replacement = '        "gpu_memory_utilization": gpu_memory_utilization,\n        "lmdeploy_model_path": lmdeploy_model_path,\n    }'
    if result_needle not in text:
        raise SystemExit('failed to update LMDeploy path in results')
    text = text.replace(result_needle, result_replacement, 1)
    changed = True

if 'choices=["hf", "vllm", "lmdeploy"]' not in text:
    engine_line = '    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf", help="Runtime engine: HuggingFace or vLLM")'
    engine_replacement = '    parser.add_argument("--engine", choices=["hf", "vllm", "lmdeploy"], default="hf", help="Runtime engine: HuggingFace, vLLM, or LMDeploy TurboMind")'
    if engine_line not in text:
        raise SystemExit('failed to locate engine argument line')
    text = text.replace(engine_line, engine_replacement, 1)
    changed = True

if 'LMDeploy cache budget' not in text:
    gpu_line = '        help="Fraction of VRAM vLLM may allocate for KV cache (vLLM engine only)",' 
    gpu_replacement = '        help="Fraction of VRAM used for KV cache (vLLM) or LMDeploy cache budget",'
    if gpu_line not in text:
        raise SystemExit('failed to update GPU memory utilization help text')
    text = text.replace(gpu_line, gpu_replacement, 1)
    changed = True

if '--lmdeploy-model-path' not in text:
    output_line = '    parser.add_argument("--output-json", help="Optional path to write JSON result")'
    injection = '    parser.add_argument("--lmdeploy-model-path", help="Path to LMDeploy TurboMind workspace when --engine lmdeploy")\n' + output_line
    if output_line not in text:
        raise SystemExit('failed to insert LMDeploy path argument')
    text = text.replace(output_line, injection, 1)
    changed = True

if 'lmdeploy_model_path=args.lmdeploy_model_path' not in text:
    call_line = '            gpu_memory_utilization=args.gpu_memory_utilization,'
    call_replacement = call_line + '\n            lmdeploy_model_path=args.lmdeploy_model_path,'
    if call_line not in text:
        raise SystemExit('failed to patch run_benchmark call')
    text = text.replace(call_line, call_replacement, 1)
    changed = True

if changed:
    path.write_text(text)
else:
    print('No changes applied; script already up to date')
PY

sudo -u azureuser python3 /tmp/patch_qwen_bench.py
