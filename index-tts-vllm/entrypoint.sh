#!/bin/bash
set -e

echo "🚀 启动 IndexTTS-2 服务..."

# 切换目录
cd /app

# 检查模型目录是否存在
if [ ! -d "./checkpoints/IndexTTS-2-vLLM" ]; then
    echo "⚠️ 未检测到模型权重，正在下载..."
    modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
fi

# 启动 WebUI
echo "🔥 启动 webui_v2.py ..."
# python3 webui_v2.py --model_dir ./checkpoints/IndexTTS-2-vLLM --host 0.0.0.0 --port 7861
python3 api_server_v2.py --model_dir ./checkpoints/IndexTTS-2-vLLM --host 0.0.0.0 --port 7861 --gpu_memory_utilization=0.5