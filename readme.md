# Qwen Benchmark Digest

This document consolidates every benchmark captured in `summarize-dec5.md`, `summarize-dec6.md`, and `summarize-dec7.md`. All runs share the Azure NC48ads_A100_v4 (2 x A100 80GB) environment unless otherwise noted. "Prompt->Gen" refers to prompt tokens requested vs. maximum requested generation tokens.

## Consolidated Results
| Date | Model | Precision | Engine / Config | Prompt->Gen (tokens) | Runtime (s) | Tok/s | Output Tokens | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-05 | Qwen2.5-14B | FP16 | HF (`device_map=auto`) | 10k->0.8k | 46.33 | 17.27 | — | Baseline Hugging Face run; see `summarize-dec5.md`. |
| 2025-12-05 | Qwen2.5-14B | INT8 | HF + bitsandbytes | 10k->0.8k | — | — | — | Failed: bitsandbytes GPU kernels missing (`summarize-dec5.md`). |
| 2025-12-05 | Qwen2.5-32B | FP16 | HF (`device_map=auto`) | 10k->0.8k | 78.21 | 10.23 | — | Pure HF baseline on dual GPUs (`summarize-dec5.md`). |
| 2025-12-05 | Qwen2.5-32B | INT8 | HF + bitsandbytes | 10k->0.8k | 299.02 | 2.68 | — | INT8 fit in <60 GB/GPU but with large throughput drop (`summarize-dec5.md`). |
| 2025-12-05 | Qwen2.5-72B | FP16 | vLLM (TP=2, FlashAttn, NVMe cache) | 10k->0.8k | 66.62 | 15.01 | — | NVMe-backed vLLM keeps 72B FP16 feasible (`summarize-dec5.md`). |
| 2025-12-05 | Qwen2.5-72B | INT8 | HF + NVMe offload | 10k->0.8k | 406.33 | 1.97 | — | Heavy NVMe reliance; INT8 still throughput-limited (`summarize-dec5.md`). |
| 2025-12-06 | Qwen2.5-32B | FP16 | vLLM (`attn=flash_attention_2`) | 20k->1k | 31.57 | 31.68 | 1,000 | Long-context vLLM run from `summarize-dec6.md`. |
| 2025-12-07 | Qwen2.5-32B | FP16 | LMDeploy TurboMind (`tp=2`, `session_len=32768`) | 20k->1k | 18.46 | 27.79 | 513 | Early EOS despite `ignore_eos`; see `summarize-dec6.md`. |
| 2025-12-06 | Qwen2.5-32B | FP16 | TensorRT-LLM (`tp=2`) | 10k->0.8k | 35.79 (net) / 68.95 (full) | 22.36 (net) / 11.60 (full) | 800 | Includes and excludes engine load time; `summarize-dec6.md`. |
| 2025-12-07 | Qwen2.5-32B | FP16 | LMDeploy TurboMind (`tp=2`) | 10k->0.8k | 15.50 | 32.25 | 500 | Shorter prompt config; see `summarize-dec7.md`. |
| 2025-12-07 | Qwen2.5-14B | FP16 | LMDeploy TurboMind (`tp=2`) | 10k->0.8k | 7.91 | 48.93 | 387 | Decode stops near 387 tokens even with `ignore_eos`; `summarize-dec7.md`. |
| 2025-12-07 | Qwen2.5-14B | FP16 | vLLM (`gpu_memory_utilization=0.9`) | 10k->0.8k | 11.99 | 66.70 | 800 | Full 800-token decode achieved; `summarize-dec7.md`. |
| 2025-12-07 | Qwen2.5-14B | FP16 | TensorRT-LLM (`tp=2`, ctx 10.8k) | 10k->0.8k | 16.93 | 47.25 | 800 | TensorRT run with flattened INT32 inputs; `summarize-dec7.md`. |

> Notes column highlights quirks (e.g., early EOS, failed runs). Refer back to each `summarize-decX.md` for expanded environment and command details.
