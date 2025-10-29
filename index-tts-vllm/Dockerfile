# ===============================
# Dockerfile for IndexTTS-2 (CUDA 12.4 + uv + 清华源)
# ===============================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# -------------------------------
# 1. 设置清华源和时区
# -------------------------------
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsm6 \
    libxext6 \
    python3 \
    python3-pip \
    && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

# -------------------------------
# 2. 配置 pip 使用清华源
# -------------------------------
RUN pip3 install --upgrade pip && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

WORKDIR /app

COPY requirements.txt requirements.txt

# -------------------------------
# 3. 使用清华源安装 Python 依赖
# -------------------------------
RUN apt-get update && apt-get install -y wget
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# COPY assets /app/assets
COPY indextts /app/indextts
COPY tools /app/tools
COPY patch_vllm.py /app/patch_vllm.py
COPY api_server.py /app/api_server.py
COPY convert_hf_format.py /app/convert_hf_format.py
COPY convert_hf_format.sh /app/convert_hf_format.sh
COPY entrypoint.sh /app/entrypoint.sh

# -------------------------------
# 4. 设置执行权限与暴露端口
# -------------------------------
RUN chmod +x /app/entrypoint.sh

EXPOSE 7861

ENTRYPOINT ["/app/entrypoint.sh"]