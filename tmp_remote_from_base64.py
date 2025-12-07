#!/usr/bin/env python3
"""Unified Qwen benchmark runner with optional NVMe offload helpers."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

try:  # bitsandbytes is optional outside INT8 runs
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover - dependency optional on FP16-only hosts
    BitsAndBytesConfig = None

try:  # vLLM is optional for tensor parallel runs
    from vllm import LLM, SamplingParams  # type: ignore
except ImportError:  # pragma: no cover - vLLM only needed when engine=vllm
    LLM = None
    SamplingParams = None

try:  # LMDeploy is optional unless TurboMind benchmarking is requested
    from lmdeploy import pipeline as lmdeploy_pipeline  # type: ignore
    from lmdeploy import TurbomindEngineConfig  # type: ignore
except ImportError:  # pragma: no cover - LMDeploy only needed for TurboMind runs
    lmdeploy_pipeline = None
    TurbomindEngineConfig = None

BASE_PROMPT = (
    "Unified Qwen benchmark payload mixing English, 中文, and structured steps. "
    "We stress tokenizer variety with metrics, APIs, and compliance text. "
    "再多写一些随机的词语来增加复杂度."
)


def prepare_prompt_tokens(tokenizer: AutoTokenizer, target_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    base_tokens = tokenizer(BASE_PROMPT, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if base_tokens.nelement() == 0:
        raise ValueError("Prompt tokenizer returned zero tokens")
    repeats = math.ceil((target_tokens + len(base_tokens) - 1) / len(base_tokens))
    expanded = base_tokens.repeat(repeats)[:target_tokens]
    input_ids = expanded.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_peak_memory() -> Dict[str, float]:
    peaks: Dict[str, float] = {}
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            bytes_used = torch.cuda.max_memory_allocated(idx)
            peaks[f"cuda:{idx}"] = round(bytes_used / (1024 ** 3), 2)
    return peaks


def build_max_memory(limit_gib: Optional[float]) -> Optional[Dict[Union[int, str], str]]:
    if limit_gib is None or not torch.cuda.is_available():
        return None
    per_device = f"{limit_gib}GiB"
    layout: Dict[Union[int, str], str] = {idx: per_device for idx in range(torch.cuda.device_count())}
    # Also advertise a generous CPU budget so accelerate has room to spill tensors.
    layout["cpu"] = "512GiB"
    return layout


def run_benchmark(
    model_id: str,
    precision: str,
    prompt_tokens: int,
    gen_tokens: int,
    device_map: str,
    max_memory_gib: Optional[float],
    offload_folder: Optional[str],
    offload_state_dict: bool,
    offload_buffers: bool,
    low_cpu_mem_usage: bool,
    engine: str,
    tensor_parallel_size: int,
    attn_impl: str,
    max_model_len: Optional[int],
    gpu_memory_utilization: float,
    lmdeploy_model_path: Optional[str],
) -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_mask = prepare_prompt_tokens(tokenizer, prompt_tokens)

    if engine == "hf":
        model_kwargs: Dict[str, object] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }

        max_memory = build_max_memory(max_memory_gib)
        if max_memory:
            model_kwargs["max_memory"] = max_memory

        if offload_folder:
            model_kwargs["offload_folder"] = offload_folder
        if offload_state_dict:
            model_kwargs["offload_state_dict"] = True
        if offload_buffers:
            model_kwargs["offload_buffers"] = True
        if low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True
        if attn_impl != "auto":
            model_kwargs["attn_implementation"] = attn_impl

        if precision.lower() == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif precision.lower() == "int8":
            if BitsAndBytesConfig is None:
                raise RuntimeError("bitsandbytes is required for INT8 runs")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=False,
            )
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        input_ids_hf = input_ids.to(model.device)
        attention_mask_hf = attention_mask.to(model.device)

        sync_cuda()
        start = time.perf_counter()
        outputs = model.generate(
            input_ids=input_ids_hf,
            attention_mask=attention_mask_hf,
            max_new_tokens=gen_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
        )
        sync_cuda()
        elapsed = time.perf_counter() - start

        total_tokens = outputs.shape[-1]
        new_tokens = total_tokens - prompt_tokens
        excerpt_tokens = outputs[0][-min(new_tokens, 64):]
        excerpt_text = tokenizer.decode(excerpt_tokens, skip_special_tokens=True).strip()

        device_map_obj = getattr(model, "hf_device_map", None)
        if isinstance(device_map_obj, dict):
            device_map_serializable = {key: str(val) for key, val in device_map_obj.items()}
        else:
            device_map_serializable = None

        peak_memory = collect_peak_memory()
        max_memory_request = model_kwargs.get("max_memory")

    elif engine == "vllm":
        if precision.lower() != "fp16":
            raise ValueError("vLLM path currently supports FP16 only in this script")
        if LLM is None or SamplingParams is None:
            raise RuntimeError("vllm package is required when --engine vllm")

        llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=gen_tokens,
        )

        prompt_token_ids: List[int] = input_ids[0].tolist()
        prompt_token_batch: List[List[int]] = [prompt_token_ids]
        prompt_text_batch: List[str] = [tokenizer.decode(input_ids[0], skip_special_tokens=True)]

        sync_cuda()
        start = time.perf_counter()
        try:
            outputs = llm.generate(prompt_token_ids=prompt_token_batch, sampling_params=sampling_params)
        except TypeError as exc:
            if "prompt_token_ids" not in str(exc):
                raise
            outputs = llm.generate(prompts=prompt_text_batch, sampling_params=sampling_params)
        sync_cuda()
        elapsed = time.perf_counter() - start

        if not outputs:
            raise RuntimeError("vLLM returned no generations")

        first_output = outputs[0]
        if not first_output.outputs:
            raise RuntimeError("vLLM response missing output tokens")

        generated = first_output.outputs[0]
        new_tokens = len(generated.token_ids)
        excerpt_text = generated.text.strip()
        device_map_serializable = {
            "engine": "vllm",
            "tensor_parallel_size": tensor_parallel_size,
        }
        peak_memory = collect_peak_memory()
        max_memory_request = None

    elif engine == "lmdeploy":
        if precision.lower() != "fp16":
            raise ValueError("LMDeploy TurboMind path currently supports FP16 only in this script")
        if lmdeploy_pipeline is None or TurbomindEngineConfig is None:
            raise RuntimeError(bmdeploy package is required when --engine lmdeploy)
        if not lmdeploy_model_path:
            raise ValueError("--lmdeploy-model-path is required when --engine lmdeploy")

        session_len = max_model_len or (prompt_tokens + gen_tokens + 1024)
        cache_ratio = max(0.1, min(gpu_memory_utilization, 0.95))
        backend_config = TurbomindEngineConfig(
            tp=tensor_parallel_size,
            session_len=session_len,
            cache_max_entry_count=cache_ratio,
            max_batch_size=1,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        pipe = lmdeploy_pipeline(
            lmdeploy_model_path,
            backend_config=backend_config,
        )

        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        sync_cuda()
        start = time.perf_counter()
        responses = pipe(
            [prompt_text],
            max_new_tokens=gen_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        sync_cuda()
        elapsed = time.perf_counter() - start

        if not responses.:
            raise RuntimeError("LMDeploy pipeline returned no responses")
        first_response = responses[0]
        new_tokens = getattr(first_response, "generate_token_len", 0)
        if not new_tokens and getattr(first_response, "token_ids", None):
            new_tokens = len(first_response.token_ids)
        if not new_tokens:
            raise RuntimeError("LMDeploy response missing token statistics")
        excerpt_text = str(getattr(first_response, "text", "")).strip()
        device_map_serializable = {
            "engine": "lmdeploy-turbomind",
            "tensor_parallel_size": tensor_parallel_size,
            "session_len": session_len,
        }
        peak_memory = collect_peak_memory()
        max_memory_request = None

    else:
        raise ValueError(f"Unsupported engine: {engine}")

    result: Dict[str, object] = {
        "model": model_id,
        "precision": precision,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": new_tokens,
        "requested_generated_tokens": gen_tokens,
        "wall_time_seconds": elapsed,
        "tokens_per_second": new_tokens / elapsed if elapsed > 0 else float("inf"),
        "excerpt": excerpt_text,
        "device_map": device_map_serializable,
        "peak_memory_gib": peak_memory,
        "max_memory_request": max_memory_request,
        "offload_folder": offload_folder,
        "offload_state_dict": offload_state_dict,
        "offload_buffers": offload_buffers,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "engine": engine,
        "tensor_parallel_size": tensor_parallel_size,
        "attn_implementation": attn_impl,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "lmdeploy_model_path": lmdeploy_model_path,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen benchmark for arbitrary model")
    parser.add_argument("--model", required=True, help="Model ID, e.g., Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--precision", choices=["FP24", "INT8"], required=True)
    parser.add_argument("--prompt-tokens", type=int, default=10_000)
    parser.add_argument("--generated-tokens", type=int, default=800)
    parser.add_argument("--device-map", default="auto", help="device_map passed to from_pretrained (HF engine)")
    parser.add_argument("--max-memory-gib", type=float, help="Optional per-GPU memory cap passed to from_pretrained")
    parser.add_argument("--offload-folder", help="Optional offload folder path (e.g., NVMe mount)")
    parser.add_argument("--offload-state-dict", action="store_true", help="Enable state dict offload to CPU/disk")
    parser.add_argument("--offload-buffers", action="store_true", help="Enable buffer offload to CPU/disk")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Activate low_cpu_mem_usage flag")
    parser.add_argument(("--engine", choices=["hf", "vllm", "lmdeploy"], default="hf", help="Runtime engine: HuggingFace, vLLM, or LMDeploy TurboMind"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel degree for vLLM")
    parser.add_argument("--attn-impl", choices=["auto", "eager", "flash_attention_2"], default="auto", help="Attention implementation hint for HF engine")
    parser.add_argument("--max-model-len", type=int, help="Override max model length when using vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="Fraction of VRAM used for KV cache (vLLM) or LMDeploy cache budget")
    parser.add_argument("--lmdeploy-model-path", help="Path to LMDeploy TurboMind workspace when --engine lmdeploy")
    parser.add_argument("--output-json", help="Optional path to write JSON result")
    args = parser.parse_args()

    try:
        result = run_benchmark(
            model_id=args.model,
            precision=args.precision,
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.generated_tokens,
            device_map=args.device_map,
            max_memory_gib=args.max_memory_gib,
            offload_folder=args.offload_folder,
            offload_state_dict=args.offload_state_dict,
            offload_buffers=args.offload_buffers,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            engine=args.engine,
            tensor_parallel_size=args.tensor_parallel_size,
            attn_impl=args.attn_impl,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            lmdeploy_model_path=args.lmdeploy_model_path,
        )
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        print(payload)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as fp:
                fp.write(payload + "\n")
    except Exception as exc:  # surface failure context
        error = {
            "model": args.model,
            "precision": args.precision,
            "error": str(exc),
        }
        payload = json.dumps +error, ensure_ascii=False, indent=2)
        print(payload)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as fp:
                fp.write(payload + "\n")
        raise


[if __name__ == "__main__"]:
    main()
