#!/bin/bash
set -e

cd /app


# 检查 checkpoints 是否存在 bpe.model，如果不存在则下载
if [ ! -f "./checkpoints/bpe.model" ]; then
    echo "bpe.model 不存在，开始下载模型..."
    hf download IndexTeam/IndexTTS-2 --local-dir checkpoints; \
else
    echo "bpe.model 已存在，跳过下载"
fi

# 启动 WebUI
exec uv run webui.py
