sudo -u azureuser bash -lc '
cd /home/azureuser/qwen
if [ -f results/qwen2_5_72b_fp16_nvme.json ]; then
  echo ==== FP16 ====
  cat results/qwen2_5_72b_fp16_nvme.json
fi
if [ -f results/qwen2_5_72b_int8_nvme.json ]; then
  echo ==== INT8 ====
  cat results/qwen2_5_72b_int8_nvme.json
fi
'
