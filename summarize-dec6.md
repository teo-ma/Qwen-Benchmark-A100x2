# Qwen2.5-32B FP16 vLLM Benchmark（2025-12-06）

## 环境概述
- Azure UK South，NC48ads_A100_v4（2 × NVIDIA A100 80GB），Ubuntu 22.04。
- Python 虚拟环境 `/home/azureuser/qwen-env`，PyTorch 2.4.0 + CUDA 12.1。
- 主要依赖：Transformers 4.57.3、vLLM 0.5.4、FlashAttention 2.6.3。
- 基准脚本：`/home/azureuser/qwen/qwen_multi_benchmark.py`（新增 vLLM `prompt_token_ids` 兼容回退逻辑）。

## 运行配置
- 模型：`Qwen/Qwen2.5-32B-Instruct`
- 精度：FP16
- 引擎：vLLM
  - `tensor_parallel_size = 2`
  - `attn_implementation = flash_attention_2`
  - `max_model_len = 32768`
  - `gpu_memory_utilization = 0.95`
- 负载：20,000 prompt tokens → 1,000 generated tokens
- 输出文件：`/home/azureuser/qwen/results/qwen2_5_32b_fp16_vllm_20k.json`

## 性能结果
| 指标 | 数值 |
| --- | --- |
| 总耗时 (s) | 31.57 |
| 吞吐 (tok/s) | 31.68 |
| 实际生成 tokens | 1,000 |

> 说明：`peak_memory_gib` 仍显示 0，因为当前统计函数无法捕获 vLLM 内部的显存分配。

## 观察
- vLLM + FlashAttention + TP=2 可在双 A100 上稳定处理 20K → 1K 的长上下文负载，并保持 >30 tok/s 的生成速度。
- 当安装的 vLLM 版本不支持 `prompt_token_ids` 参数时，脚本会自动退回到 `prompts=str` 模式，避免版本差异导致的报错。

## Qwen2.5-32B FP16 LMDeploy Benchmark（2025-12-07）

### 环境概述
- 同一 Azure UK South 环境（NC48ads_A100_v4，2 × A100 80GB），Python 虚拟环境 `/home/azureuser/qwen-env`。
- 依赖新增 LMDeploy 0.5.2（TurboMind backend），沿用 PyTorch 2.3.1/cu121 与 TensorRT-LLM 0.11.0。
- Turbomind workspace：`/home/azureuser/qwen/lmdeploy/qwen2_5_32b_tp2`（`tp=2` 转换产物）。
- 基准脚本：`/home/azureuser/qwen/qwen_multi_benchmark.py`（已合并 LMDeploy 分支与 CLI 选项）。

### 运行配置
- 模型：`Qwen/Qwen2.5-32B-Instruct`
- 精度：FP16
- 引擎：LMDeploy TurboMind
  - `tensor_parallel_size = 2`
  - `session_len = 32768`
  - `gpu_memory_utilization = 0.90`（控制 KV cache 占比）
- 负载：20,000 prompt tokens → 1,000 请求 tokens（实际生成 513 tokens）
- 输出文件：`/home/azureuser/qwen/results/qwen2_5_32b_fp16_lmdeploy_20k.json`

### 性能结果
| 指标 | 数值 |
| --- | --- |
| 总耗时 (s) | 18.46 |
| 吞吐 (tok/s) | 27.79 |
| 实际生成 tokens | 513 |

> 说明：TurboMind 引擎在 513 token 处提前遇到 EOS，因此 `requested_generated_tokens=1000` 未完全使用；`peak_memory_gib` 统计同样因外部监控限制显示为 0。

### 观察
- TurboMind 在相同期望负载下可提供 ~27.8 tok/s 的生成速度，比 TensorRT-LLM（含加载后净 22 tok/s）更快，但略低于 vLLM 的 31 tok/s。
- TurboMind 结果对 GEMM autotune 配置不敏感；`gemm_config.in` 缺失时使用默认算法即可维持稳定性能。
- 本地脚本新增 `--engine lmdeploy`、`--lmdeploy-model-path` 等选项后，可统一驱动 HF/vLLM/LMDeploy 三条路径，后续再跑其他组合仅需切换 CLI。

## Qwen2.5-32B FP16 TensorRT-LLM Benchmark（2025-12-06）

### 环境概述
- 同一 Azure UK South 环境（NC48ads_A100_v4，2 × A100 80GB），Python 虚拟环境 `/home/azureuser/qwen-env`。
- TensorRT-LLM 0.11.0 + TensorRT 10.1.0（CUDA 12.1），PyTorch 2.3.1/cu121，含 NCCL/OpenMPI 依赖。
- 引擎产物：`/home/azureuser/qwen/trt_engines/qwen2_5_32b_fp16_tp2_ctx10800`（`--tp_size 2`，`--max_input_len 10000`，`--max_seq_len 10800`，`--gemm_plugin float16`，`--gpt_attention_plugin float16`）。

### 运行配置
- 模型：`Qwen/Qwen2.5-32B-Instruct`，精度 FP16，TensorRT-LLM 引擎，TP=2。
- Prompt：使用脚本生成的 `input_prompt_10k.npy`（10,000 tokens），生成上限 800 tokens。
- 命令：
  ```bash
  cd /home/azureuser/TensorRT-LLM/examples
  source /home/azureuser/qwen-env/bin/activate
  /usr/bin/time -f 'ELAPSED_SEC %e' \ 
    mpirun -n 2 python run.py \
      --engine_dir /home/azureuser/qwen/trt_engines/qwen2_5_32b_fp16_tp2_ctx10800 \
      --tokenizer_dir /home/azureuser/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
      --input_file /home/azureuser/qwen/input_prompt_10k.npy \
      --max_input_length 10000 --max_output_len 800 \
      --output_npy /home/azureuser/qwen/output_trt.npy --log_level info
  ```

### 性能结果
| 指标 | 数值 |
| --- | --- |
| 总耗时（含加载） | 68.95 s |
| 引擎加载时间（单 rank 日志） | 33.16 s |
| 推理耗时（68.95 − 33.16） | 35.79 s |
| 生成吞吐（净推理） | 22.36 tok/s |
| 生成吞吐（含加载） | 11.60 tok/s |
| 实际生成 tokens | 800 |
| 输出文件 | `/home/azureuser/qwen/output_trt.npy` |

### 观察
- 首次请求需要 ~33 s 将 32B TP2 引擎载入两块 A100，之后的推理阶段耗时 ~35.8 s，可提供 ~22 tok/s 的净生成速度（10k prompt → 0.8k output）。
- `output_trt.npy` 中的序列长度为 10,800，按 tokenizer `pad_id=151643` 统计，实际新增 800 tokens，与目标配置一致。
- 由于 `run.py` 每次都会重新加载引擎，此处同时给出含加载与剔除加载的吞吐指标，方便与 vLLM 结果对比。
