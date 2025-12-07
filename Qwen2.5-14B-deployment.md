# Qwen2.5-14B 在 Azure Spot GPU VM 的部署与测试指南

## 环境与资源
- **区域**：UK South (`uksouth`)
- **资源组**：`rg-qwen-uksouth`
- **网络**：虚拟网络 `vnet-qwen`（地址段 `10.10.0.0/16`），子网 `snet-gpu`
- **安全**：NSG `nsg-gpu`，放行 TCP/22；加速网卡 `nic-gpu` 绑定静态公网 IP `pip-gpu`（`51.142.202.62`）
- **计算**：Spot VM `vm-qwen-nc48`，规格 `Standard_NC48ads_A100_v4`，Ubuntu 22.04，1 TB Premium_LRS OS 磁盘
- **认证**：SSH 密钥 `~/.ssh/qwen_uksouth_rsa`，登录命令 `ssh -i ~/.ssh/qwen_uksouth_rsa azureuser@51.142.202.62`

## 部署步骤
1. **登录与配额检查**
   - `az login`
   - 使用 `appmod-check-quota` 或 `az vm list-usage -l uksouth` 确认 `Standard_NC48ads_A100_v4` 可用配额。
2. **网络与安全**
   ```bash
   az group create -n rg-qwen-uksouth -l uksouth
   az network vnet create -g rg-qwen-uksouth -n vnet-qwen \
     --address-prefix 10.10.0.0/16 --subnet-name snet-gpu --subnet-prefix 10.10.1.0/24
   az network nsg create -g rg-qwen-uksouth -n nsg-gpu -l uksouth
   az network nsg rule create -g rg-qwen-uksouth --nsg-name nsg-gpu \
     -n AllowSSH --priority 1000 --direction Inbound --access Allow \
     --protocol Tcp --destination-port-range 22 --source-address-prefixes "*"
   az network public-ip create -g rg-qwen-uksouth -n pip-gpu --sku Standard --allocation-method Static
   az network nic create -g rg-qwen-uksouth -n nic-gpu --subnet snet-gpu --vnet-name vnet-qwen \
     --network-security-group nsg-gpu --public-ip-address pip-gpu
   az network nic update -g rg-qwen-uksouth -n nic-gpu --accelerated-networking true
   ```
3. **Spot GPU VM**
   ```bash
   az vm create -g rg-qwen-uksouth -n vm-qwen-nc48 -l uksouth \
     --nics nic-gpu --image Ubuntu2204 --size Standard_NC48ads_A100_v4 \
     --os-disk-size-gb 1024 --storage-sku Premium_LRS \
     --priority Spot --max-price -1 --eviction-policy Deallocate \
     --admin-username azureuser --ssh-key-values ~/.ssh/qwen_uksouth_rsa.pub \
     --enable-secure-boot false --enable-vtpm false
   ```
   > Spot VM 可能被回收，可用 `az vm start` 恢复；若需稳定运行，可改为按需实例。
4. **GPU 驱动**
   ```bash
   az vm extension set -g rg-qwen-uksouth --vm-name vm-qwen-nc48 \
     --publisher Microsoft.HpcCompute --name NvidiaGpuDriverLinux \
     --version 1.11 --settings '{"updateSettings":"INSTALL","driverVersion":"cuda_12.2"}'
   ```
5. **Python 推理环境**
   ```bash
   az vm run-command invoke -g rg-qwen-uksouth -n vm-qwen-nc48 --command-id RunShellScript --scripts \
     "sudo apt-get update" \
     "sudo apt-get install -y python3-venv python3-pip git" \
     "sudo -u azureuser bash -c 'cd ~ && python3 -m venv qwen-env && source qwen-env/bin/activate && \
        pip install --upgrade pip && \
        pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 && \
        pip install transformers==4.44.2 accelerate==0.30.1 sentencepiece protobuf==4.25.3 numpy einops \
                   huggingface_hub[cli]==0.25.2 bitsandbytes==0.43.1'"
   ```
6. **推理脚本（冒烟测试）**
   - 文件：`/home/azureuser/qwen/run_qwen_inference.py`
   - 内容：载入 `Qwen/Qwen2.5-14B-Instruct`（FP16, `device_map='auto'`），执行中文问答并将结果写入 `~/qwen/last_run.log`。
   - 运行命令：
     ```bash
     az vm run-command invoke -g rg-qwen-uksouth -n vm-qwen-nc48 --command-id RunShellScript --scripts \
       "sudo -u azureuser bash -c 'source ~/qwen-env/bin/activate && python ~/qwen/run_qwen_inference.py > ~/qwen/last_run.log'"
     ```

## 测试场景与结果
### 1. 功能冒烟测试
- **目标**：验证模型载入、FP16 推理链路和示例中文问答。
- **结果**：脚本输出描述 Qwen2.5-14B 的特点并回答“如何学习 Python 编程”，表明模型正常工作。
- **注意**：首次运行需下载 ~8 个 shard（约 2–3 分钟）；缓存保留在 OS 磁盘。

### 2. 长上下文性能测试
- **脚本**：`/home/azureuser/qwen/run_qwen_benchmark.py`
- **场景**：构造 10,000 token 输入（重复中英段落），生成 800 token 输出，`do_sample=False`（deterministic）。
- **执行**：
  ```bash
  az vm run-command invoke -g rg-qwen-uksouth -n vm-qwen-nc48 --command-id RunShellScript --scripts \
    "sudo -u azureuser bash -c 'source ~/qwen-env/bin/activate && python ~/qwen/run_qwen_benchmark.py | tee ~/qwen/benchmark_10k_prompt.log'"
  ```
- **结果日志**（`benchmark_10k_prompt.log`）：
  ```json
  {
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "prompt_tokens": 10000,
    "generated_tokens": 800,
    "wall_time_seconds": 48.15,
    "tokens_per_second": 16.61,
    "device_type": "cuda:0"
  }
  ```
- **结论**：生成阶段吞吐约 16.6 token/s，满足 10k+ 上下文场景的性能需求。如需采样，可将 `do_sample` 设为 `True` 并调整温度/Top-p。

## 运行与维护
- **SSH 登录**：`ssh -i ~/.ssh/qwen_uksouth_rsa azureuser@51.142.202.62`
- **Spot VM 管理**：若 `powerState=deallocated`，使用 `az vm start`；必要时切换为按需实例。
- **日志位置**：
  - 冒烟结果：`~/qwen/last_run.log`
  - 性能测试：`~/qwen/benchmark_10k_prompt.log`
- **清理命令**：完成测试后按顺序 `az vm delete`、删除 NIC/Public IP/NSG/VNet，最后 `az group delete`，避免 GPU 费用。

## 后续建议
1. 将脚本封装为 API 服务（FastAPI + Uvicorn）或使用 Azure Container Apps 提供远程推理接口。
2. 使用 Azure Monitor 或自定义脚本跟踪 Spot 驱逐事件，必要时自动重启并预热模型缓存。
3. 若需持续基准，可将 `run_qwen_benchmark.py` 纳入 CI/CD 流程，并记录不同 prompt 长度的吞吐曲线。
