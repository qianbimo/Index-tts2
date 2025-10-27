# ================================
# Dockerfile for IndexTTS-2-vLLM
# CUDA 12.4 + Ubuntu 22.04 + uv + 清华镜像
# ================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---------------------
# 基础环境设置
# ---------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app

# ---------------------
# 安装系统依赖
# ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs python3 python3-pip ffmpeg libsndfile1 curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------
# 复制项目文件
# ---------------------
COPY . /app

# ---------------------
# 安装 uv 包管理器
# ---------------------
RUN pip install --upgrade pip && pip install uv

# ---------------------
# 使用 uv 安装 Python 项目依赖（清华镜像）
# 注意：加上 --system 避免虚拟环境报错
# ---------------------
RUN uv pip install --system --upgrade pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    uv pip install --system -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# ---------------------
# 安装 HuggingFace CLI（用于下载模型）
# ---------------------
RUN uv tool install "huggingface-hub[cli,hf_xet]"

# ---------------------
# 确保 entrypoint 可执行
# ---------------------
RUN chmod +x /app/entrypoint.sh

# ---------------------
# 暴露端口
# ---------------------
EXPOSE 7861

# ---------------------
# 默认启动命令
# ---------------------
ENTRYPOINT ["/app/entrypoint.sh"]
