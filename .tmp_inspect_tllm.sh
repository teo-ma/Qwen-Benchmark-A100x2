sudo -u azureuser bash -lc '
source /home/azureuser/qwen-env/bin/activate
python3 -c "import os, tensorrt_llm; print(tensorrt_llm.__file__); print(os.listdir(os.path.dirname(tensorrt_llm.__file__)))"
'
