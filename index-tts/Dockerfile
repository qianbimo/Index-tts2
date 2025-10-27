# ============================================
# IndexTTS2 Dockerfile (CUDA 12.4 + GPU, Optimized for China)
# Author: ChatGPT
# ============================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ---------------------
# 基础环境设置
# ---------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_ENDPOINT="https://hf-mirror.com"
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app

# ---------------------
# 安装系统依赖
# ---------------------
RUN apt-get update && apt-get install -y \
    git git-lfs python3 python3-pip ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------
# 克隆 IndexTTS2 仓库
# ---------------------
COPY . /app

# ---------------------
# 安装 Python 包管理器 uv
# ---------------------
RUN pip install -U pip uv

# ---------------------
# 安装项目依赖（仅启用 WebUI，不包含 DeepSpeed）
# 使用清华镜像加速
# ---------------------
RUN uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# ---------------------
# 安装 HuggingFace CLI（用于下载模型）
# ---------------------
RUN uv tool install "huggingface-hub[cli,hf_xet]"

# ---------------------
# 暴露 Web 端口
# ---------------------
EXPOSE 7860

# ---------------------
# 默认启动命令（运行 Gradio Web 界面）
# ---------------------
ENTRYPOINT ["/app/entrypoint.sh"]
