# Qwen Benchmark Digest

This document consolidates every benchmark captured in `summarize-dec5.md`, `summarize-dec6.md`, and `summarize-dec7.md`. All runs share the Azure NC48ads_A100_v4 (2 x A100 80GB) environment unless otherwise noted. "Prompt->Gen" refers to prompt tokens requested vs. maximum requested generation tokens.

## FP16 Benchmark Comparison（按模型大小）

### Qwen2.5-72B
| Date | Engine / Config | Prompt->Gen (tokens) | Runtime (s) | Tok/s | Output Tokens | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-12-05 | vLLM (TP=2, FlashAttn, NVMe cache) | 10k->0.8k | 66.62 | 15.01 | — | NVMe-backed vLLM keeps 72B FP16 feasible；详情见 `summarize-dec5.md` |

### Qwen2.5-32B
| Date | Engine / Config | Prompt->Gen (tokens) | Runtime (s) | Tok/s | Output Tokens | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-12-05 | HF (`device_map=auto`) | 10k->0.8k | 78.21 | 10.23 | — | 纯 Hugging Face 基线；参见 `summarize-dec5.md` |
| 2025-12-06 | vLLM (`attn=flash_attention_2`) | 20k->1k | 31.57 | 31.68 | 1,000 | 长上下文 vLLM 运行；参见 `summarize-dec6.md` |
| 2025-12-07 | LMDeploy TurboMind (`tp=2`, `session_len=32768`) | 20k->1k | 18.46 | 27.79 | 513 | `ignore_eos` 后仍在 513 token 提前停止；参见 `summarize-dec6.md` |
| 2025-12-06 | TensorRT-LLM (`tp=2`) | 10k->0.8k | 35.79 (净推理) / 68.95 (含加载) | 22.36 (净) / 11.60 (含加载) | 800 | 同时列出剔除/包含引擎加载时间；参见 `summarize-dec6.md` |
| 2025-12-07 | LMDeploy TurboMind (`tp=2`) | 10k->0.8k | 15.50 | 32.25 | 500 | 缩短 prompt，解码在 500 token 停止；参见 `summarize-dec7.md` |

### Qwen2.5-14B
| Date | Engine / Config | Prompt->Gen (tokens) | Runtime (s) | Tok/s | Output Tokens | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-12-05 | HF (`device_map=auto`) | 10k->0.8k | 46.33 | 17.27 | — | 14B FP16 Hugging Face 基线；参见 `summarize-dec5.md` |
| 2025-12-07 | LMDeploy TurboMind (`tp=2`) | 10k->0.8k | 7.91 | 48.93 | 387 | `--ignore-eos` 后仍在 387 token 提前停止；参见 `summarize-dec7.md` |
| 2025-12-07 | vLLM (`gpu_memory_utilization=0.9`) | 10k->0.8k | 11.99 | 66.70 | 800 | 成功生成完整 800 token；参见 `summarize-dec7.md` |
| 2025-12-07 | TensorRT-LLM (`tp=2`, ctx 10.8k) | 10k->0.8k | 16.93 | 47.25 | 800 | 需将输入改为逐样本 1-D INT32；参见 `summarize-dec7.md` |

## INT8 Benchmark Snapshot
| Date | Model | Engine / Config | Prompt->Gen (tokens) | Runtime (s) | Tok/s | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-12-05 | Qwen2.5-14B | HF + bitsandbytes | 10k->0.8k | — | — | 失败：bitsandbytes 缺 GPU kernel 与 `triton.ops`；`summarize-dec5.md` |
| 2025-12-05 | Qwen2.5-32B | HF + bitsandbytes | 10k->0.8k | 299.02 | 2.68 | INT8 将显存压到 <60 GB/GPU，但吞吐下降显著；`summarize-dec5.md` |
| 2025-12-05 | Qwen2.5-72B | HF + NVMe offload | 10k->0.8k | 406.33 | 1.97 | 借助 NVMe 才能跑通，性能受限；`summarize-dec5.md` |

> 同一模型的结果已聚合在各自表格中，方便对比不同推理引擎；如需更详细的运行日志、命令与环境，请参阅对应的 `summarize-decX.md`。
