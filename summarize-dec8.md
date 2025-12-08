# Qwen Benchmark Log – 2025-12-08

## DeepSpeed 安装与 DNS 修复
- 运行 `az vm run-command ... cat /etc/resolv.conf` 发现符号链接指向不存在的 `../run/systemd/resolve/stub-resolv.conf`，导致 pip 无法解析域名（`Temporary failure in name resolution`）。
- 通过 `sudo tee /etc/resolv.conf <<<"nameserver 168.63.129.16"` 将解析服务器指向 Azure 默认 DNS，随后 `nslookup pypi.org` 恢复正常。
- 在 `~/qwen-env` 中执行 `pip install --upgrade deepspeed`，成功构建 0.18.2 版本；警告仅与已有 CUDA wheel 命名有关，对推理无影响。
- 编写 `scripts/run_deepspeed_benchmark_14b_10k.sh`，统一 torchrun 启动参数：`--nproc_per_node=2 --master_port 29525`，并将结果写入 `~/qwen/results/qwen2_5_14b_fp16_deepspeed_10k.json`。

## Qwen2.5-14B FP16 DeepSpeed Benchmark（10K → 0.8K）
- 环境：`~/qwen-env`（PyTorch 2.3.1 + CUDA 12.1），双 A100 80GB；`CUDA_VISIBLE_DEVICES=0,1`。
- 命令：
  ```bash
  scripts/run_deepspeed_benchmark_14b_10k.sh
  ```
- 关键日志：DeepSpeed 在初始化阶段提示 `replace_method` / `mp_size` 已废弃，但自动回落到新的 `tensor_parallel` 配置；`torch_dtype` 警告来自 Transformers 近期参数重命名，可忽略。

### 性能结果（取 rank0 JSON）
| 指标 | 数值 |
| --- | --- |
| `wall_time_seconds` | 53.92 s |
| `tokens_per_second` | 14.84 tok/s |
| `generated_tokens` | 800 |
| `peak_memory_gib` | cuda:0 → 30.66 / cuda:1 → 0.00 |
| `distributed_world_size` | 2 |

### 观察
- 该运行完整生成 800 token，但解码期间 DeepSpeed 忽略 `temperature/top_p`（日志提示这类 flag 在 inference mode 下无效），推理路径等价于贪心解码。
- 显存占用集中在 `cuda:0`，`cuda:1` 因张量切分仅承载激活与少量权重；如需更均衡可改用 `tensor_parallel.tp_size=2` 明确注入。
- 相比 12 月 7 日的 vLLM（11.99 s / 66.7 tok/s），DeepSpeed 版本延迟更高但实现路径简单，可作为 Hugging Face 原生的多卡基线；数据已汇入 `summarize-dec8.md` 并用于 README 表格。

## Qwen2.5-32B FP16 DeepSpeed Benchmark（10K → 0.8K）
- 环境与 14B 相同：`~/qwen-env` + 2 × A100 80GB；使用脚本 `scripts/run_deepspeed_benchmark_32b_10k.sh`（`MASTER_PORT=29535`）。
- 运行命令：
  ```bash
  scripts/run_deepspeed_benchmark_32b_10k.sh
  ```
- DeepSpeed 仍提示 `replace_method/mp_size` 已废弃，同时在加载 32B 权重时 `torch_dtype` 重复警告；除此之外运行稳定。

### 性能结果（取 rank0 JSON）
| 指标 | 数值 |
| --- | --- |
| `wall_time_seconds` | 83.83 s |
| `tokens_per_second` | 9.54 tok/s |
| `generated_tokens` | 800 |
| `peak_memory_gib` | cuda:0 → 65.42 / cuda:1 → 0.00 |
| `distributed_world_size` | 2 |

### 观察
- 32B DeepSpeed 完整生成 800 token，但总耗时（83.8 s）略高于 12 月 5 日 HF baseline（78.2 s），推测原因是无量化/分片 offload，所有权重集中在 rank0 并触发更多通信。
- `peak_memory_gib` 显示 `cuda:1` 为 0.00，原因与 14B 相同：第二张卡只承载轻量激活，峰值不足 0.01 GiB；若需均衡显存，可在 DeepSpeed 配置中显式启用 `tensor_parallel.tp_size=2`。
- 尽管吞吐（9.5 tok/s）低于 TurboMind/vLLM，DeepSpeed 仍提供一个无需额外引擎转换的 HF 原生多卡路径，后续可考虑结合 ZeRO-Inference 或权重并行以提升性能。
