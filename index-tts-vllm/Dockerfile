# ===============================
# Dockerfile for IndexTTS-2 (CUDA 12.4 + uv + 清华源)
# ===============================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 基本环境配置
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:$PATH"

# -------------------------------
# 1. 配置 APT 清华源
# -------------------------------
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 安装系统依赖与 Python3
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3.10-dev \
    build-essential git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 2. 配置 pip 清华源和信任主机
# -------------------------------
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip config set global.timeout 120

# 安装 uv（使用清华源）
RUN pip install --no-cache-dir uv -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 配置 uv 使用清华源（修复目录不存在问题）
RUN mkdir -p /root/.config/uv && \
    echo -e "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n" > /root/.config/uv/pyproject.toml

# 设置工作目录
WORKDIR /app

# 复制项目代码
COPY assets /app/assets
COPY indextts /app/indextts
COPY tools /app/tools
COPY patch_vllm.py /app/patch_vllm.py
COPY api_server_v2.py /app/api_server_v2.py
COPY convert_hf_format.py /app/convert_hf_format.py
COPY convert_hf_format.sh /app/convert_hf_format.sh
COPY entrypoint.sh /app/entrypoint.sh
COPY requirements.txt /app/requirements.txt

# -------------------------------
# 3. 使用 PyTorch 官方 CUDA 12.4 源安装 PyTorch 2.6.0
# -------------------------------
RUN uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# -------------------------------
# 4. 安装项目依赖（使用 uv + 清华源）
# -------------------------------
RUN cd /app && uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# -------------------------------
# 5. 安装 modelscope（使用清华源）
# -------------------------------
RUN uv pip install modelscope

# -------------------------------
# 6. 预下载 IndexTTS-2 模型权重
# -------------------------------
RUN mkdir -p checkpoints && \
    modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM || true

# -------------------------------
# 7. 设置执行权限与暴露端口
# -------------------------------
RUN chmod +x /app/entrypoint.sh /app/convert_hf_format.sh

EXPOSE 7861

ENTRYPOINT ["/app/entrypoint.sh"]