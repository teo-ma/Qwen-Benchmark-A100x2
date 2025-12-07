# Qwen 推理测试汇总（2025-12-05）

## 测试背景
- 区域/算力：Azure UK South，NC48ads_A100_v4（2 x A100 80GB），Ubuntu 22.04。
- 软件栈：Python venv `/home/azureuser/qwen-env`，PyTorch 2.4.0，Transformers 4.57.3，vLLM 0.5.4，FlashAttention 2.6.3。
- 场景：统一 10k prompt tokens / 0.8k generated tokens 的推理负载，比较 Qwen2.5 14B/32B/72B 在 FP16 与 INT8 下的表现。

## 性能对比表
| 模型 | 精度 | 引擎/配置 | 运行时 (s) | 吞吐 (tok/s) | 备注 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-14B | FP16 | HF, `device_map=auto` | 46.33 | 17.27 | 纯 GPU，稳定通过 |
| Qwen2.5-14B | INT8 | HF + bitsandbytes | — | — | 失败：bitsandbytes 缺 GPU 支持且缺 `triton.ops` |
| Qwen2.5-32B | FP16 | HF, `device_map=auto` | 78.21 | 10.23 | GPU 双卡切分，无需 NVMe |
| Qwen2.5-32B | INT8 | HF + bitsandbytes | 299.02 | 2.68 | 显存降至 <60GB/GPU，吞吐下降明显 |
| Qwen2.5-72B | FP16 | vLLM, TP=2, FlashAttn, NVMe cache | 66.62 | 15.01 | 借助 vLLM + NVMe 才能跑通，性能优于 32B FP16 |
| Qwen2.5-72B | INT8 | HF + NVMe offload | 406.33 | 1.97 | 通过 NVMe 扩展容量，吞吐受限 |

## 测试级别说明
1. **基础级 (14B FP16)**：验证脚本、环境与基线性能。
2. **扩展级 (32B FP16/INT8)**：考察更大模型在相同算力上的可行性与 INT8 带来的显存收益。
3. **高阶级 (72B FP16/INT8)**：引入 NVMe 缓存、vLLM Tensor Parallel、FlashAttention 等优化以撑起超大模型，评估性能与代价。

## 结果总结
- vLLM + TP2 + FlashAttention 是 72B FP16 成功与高吞吐的关键，使其以 66.62s/15.01 tok/s 超过 32B FP16。
- INT8 虽显著降低显存占用，但在 32B/72B 实测吞吐分别下降 ~74% / ~87%，仅在容量受限时值得启用。
- 14B INT8 需先重新安装 GPU 版 bitsandbytes（例如 `pip install --no-cache-dir bitsandbytes==0.44.1 triton==3.0.0`）后重测，以完成矩阵。