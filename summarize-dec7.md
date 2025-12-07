# Qwen2.5 LMDeploy Benchmarks（2025-12-07）

## 环境概述
- Azure UK South，NC48ads_A100_v4（2 × NVIDIA A100 80GB），Ubuntu 22.04。
- Python 虚拟环境 `/home/azureuser/qwen-env`，PyTorch 2.3.1 + CUDA 12.1。
- 另建 vLLM 专用虚拟环境 `/home/azureuser/vllm-env`（PyTorch 2.9.0 + CUDA 12.8，vLLM 0.12.0），避免破坏 LMDeploy 依赖树；`scripts/run_vllm_benchmark_14b_10k.sh` 通过 `VLLM_ENV_PATH` 切换该环境。
- 依赖：Transformers 4.57.3、TensorRT-LLM 0.11.0、LMDeploy 0.5.2（TurboMind backend）。
- 基准脚本：`/home/azureuser/qwen/qwen_multi_benchmark.py`（统一 HF/vLLM/LMDeploy 入口）。

## 运行配置（10K → 0.8K）
- 模型：`Qwen/Qwen2.5-32B-Instruct`，精度 FP16。
- 引擎：LMDeploy TurboMind，`tensor_parallel_size = 2`，`session_len = 32768`（沿用既有 workspace）。
- 负载：10,000 prompt tokens → 800 请求 tokens。
- 运行命令：`scripts/run_lmdeploy_benchmark_10k.sh`（内部执行 `python ~/qwen/qwen_multi_benchmark.py --prompt-tokens 10000 --generated-tokens 800 --engine lmdeploy ...`）。
- 输出文件：`/home/azureuser/qwen/results/qwen2_5_32b_fp16_lmdeploy_10k.json`。

## 性能结果
| 指标 | 数值 |
| --- | --- |
| 总耗时 (s) | 15.50 |
| 吞吐 (tok/s) | 32.25 |
| 实际生成 tokens | 500 |

> 说明：TurboMind 在 500 token 处命中 EOS，因此未完全使用 800 token 的预算；`peak_memory_gib` 统计仍为 0（同上一版脚本限制）。

## 对比（20K → 1K vs. 10K → 0.8K）
| 负载 | Prompt Tokens | 生成请求 | 实际生成 | 耗时 (s) | 吞吐 (tok/s) |
| --- | --- | --- | --- | --- | --- |
| 20K → 1K | 20,000 | 1,000 | 513 | 18.46 | 27.79 |
| 10K → 0.8K | 10,000 | 800 | 500 | 15.50 | 32.25 |

- Prompt 减半后，TurboMind 吞吐提升至 ~32 tok/s，且总耗时缩短 3 s。
- 较短上下文仍会提前触发 EOS，可考虑调高 `session_len` 或通过 LMDeploy pipeline 配置停止词（禁用默认 EOS）以逼近完整 800 token 生成。
- GEMM autotune 依旧不是瓶颈；运行日志仅提示 `gemm_config.in` 缺失，采用默认算法即可。

## Qwen2.5-14B FP16 LMDeploy Benchmark（10K → 0.8K）
- Workspace：`/home/azureuser/qwen/lmdeploy/qwen2_5_14b_tp2`（通过 `scripts/convert_lmdeploy_14b.sh` 以 TP=2 转换）。
- 运行命令：`scripts/run_lmdeploy_benchmark_14b_10k.sh`（内部沿用统一 `qwen_multi_benchmark.py` 参数集）。
- 输出文件：`/home/azureuser/qwen/results/qwen2_5_14b_fp16_lmdeploy_10k.json`。

### 性能结果
| 指标 | 数值 |
| --- | --- |
- 总耗时 (s) | 7.91 |
- 吞吐 (tok/s) | 48.93 |
- 实际生成 tokens | 387 |

- 最新一次运行仍使用 `--ignore-eos`，但 TurboMind 在 387 token 左右触发停止（日志显示 `stop_words` 已为空，推测 pipeline 内置的 repetition 规则或 session 长度限制所致）。
- 在 decode 更早终止的情况下，总耗时缩短到 ~7.9 s，不过平均吞吐回落至 ~49 tok/s；后续如需对标 vLLM，可继续排查 LMDeploy 停止条件以争取完整 800 token。
- `peak_memory_gib` 依旧为 0，之后可仿照 vLLM 分支在脚本内主动查询 `torch.cuda.max_memory_allocated()` 以完善统计。

## Qwen2.5-14B FP16 vLLM Benchmark（10K → 0.8K）
- 运行环境：`/home/azureuser/vllm-env`（torch 2.9.0、vLLM 0.12.0），通过 `scripts/run_vllm_benchmark_14b_10k.sh` 激活（可用 `VLLM_ENV_PATH=/path/to/env` 覆盖）。
- 启动命令：`python ~/qwen/qwen_multi_benchmark.py --engine vllm --prompt-tokens 10000 --generated-tokens 800 --tensor-parallel-size 2 --max-model-len 20480 --gpu-memory-utilization 0.9`。
- 输出文件：`/home/azureuser/qwen/results/qwen2_5_14b_fp16_vllm_10k.json`。

### 性能结果
| 指标 | 数值 |
| --- | --- |
| 总耗时 (s) | 11.99 |
| 吞吐 (tok/s) | 66.70 |
| 实际生成 tokens | 800 |

- vLLM 解码未再触发 EOS，成功生成完整 800 token，吞吐比 TurboMind 版本高 ~9%，整体延迟压缩到 ~12 s。
- `peak_memory_gib` 仍返回 0，后续可在 vLLM 分支补充显存采样逻辑或改为解析 vLLM 的 `--gpu-memory-utilization` telemetry。
- 运行日志显示 Prefill ≈ 835 tok/s、Decode ≈ 66 tok/s，如需更高吞吐可尝试 `gpu_memory_utilization`=0.95 或引入 `paged_attention`。

## Qwen2.5-14B FP16 TensorRT-LLM Benchmark（10K → 0.8K）
- 引擎：TensorRT-LLM 0.11.0，使用已有 `~/qwen/trt_engines/qwen2_5_14b_fp16_tp2_ctx10800`（TP=2、上下文 10.8k）并通过 `mpirun -n 2` 驱动 `qwen_multi_benchmark.py --engine trtllm`。
- 运行命令写入 `~/qwen/results/qwen2_5_14b_fp16_trtllm_10k.json`，仅 rank 0 输出结果，非 root 进程静默退出避免重复日志。

### 性能结果
| 指标 | 数值 |
| --- | --- |
| 总耗时 (s) | 16.93 |
| 吞吐 (tok/s) | 47.25 |
| 实际生成 tokens | 800 |
| 峰值显存 (GiB) | cuda:0 → 0.99 / cuda:1 → 0.00 |

- 输入张量改为逐样本 1-D INT32，否则 TensorRT runtime 在分割 10k token 长序列时会抛出 `split_with_sizes` 错误。
- 本次运行成功生成完整 0.8k token，与 TurboMind/vLLM 相比，延迟介于两者之间；TensorRT 是唯一返回显存统计的路径，可作为后续显存监控参考。
