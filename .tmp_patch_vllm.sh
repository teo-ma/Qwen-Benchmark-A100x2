sudo -u azureuser python3 - <<'PY'
from pathlib import Path
path = Path('/home/azureuser/qwen/qwen_multi_benchmark.py')
text = path.read_text()
old = """        llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=\"float16\",
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

        sync_cuda()
        start = time.perf_counter()
        outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        sync_cuda()
        elapsed = time.perf_counter() - start
"""
new = """        llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=\"float16\",
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
            if \"prompt_token_ids\" not in str(exc):
                raise
            outputs = llm.generate(prompts=prompt_text_batch, sampling_params=sampling_params)
        sync_cuda()
        elapsed = time.perf_counter() - start
"""
if old not in text:
    raise SystemExit('target snippet not found')
path.write_text(text.replace(old, new, 1))
PY
