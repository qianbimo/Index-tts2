#!/usr/bin/env bash
set -e

# ====================================================
# IndexTTS-2 vLLM Docker 启动脚本 (CUDA 12.4 + uv 优化版)
# ====================================================

# ---------------- 参数与默认值 ----------------
MODEL_DIR=${MODEL_DIR:-"/app/checkpoints"}
MODEL=${MODEL:-"IndexTeam/IndexTTS-2-vLLM"}
VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-1}
DOWNLOAD_MODEL=${DOWNLOAD_MODEL:-1}
CONVERT_MODEL=${CONVERT_MODEL:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.25}
PORT=${PORT:-7861}
PYPI_MIRROR=${PYPI_MIRROR:-"https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"}

echo "===================================================="
echo "🚀 启动 IndexTTS-2 vLLM 服务"
echo "📁 模型目录: $MODEL_DIR"
echo "🧠 模型名称: $MODEL"
echo "🌐 使用 ModelScope: $VLLM_USE_MODELSCOPE"
echo "⬇️ 允许下载模型: $DOWNLOAD_MODEL"
echo "🔄 转换模型: $CONVERT_MODEL"
echo "🧮 GPU 显存占比: $GPU_MEMORY_UTILIZATION"
echo "🌍 服务端口: $PORT"
echo "📦 PyPI 镜像源: $PYPI_MIRROR"
echo "===================================================="

# ---------------- 函数定义 ----------------
log() { echo -e "\033[1;32m[INFO]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }

check_model_exists() {
    if [ ! -d "$MODEL_DIR" ]; then
        warn "模型目录 $MODEL_DIR 不存在"
        return 1
    fi
    if [ ! -f "$MODEL_DIR/.download_complete" ]; then
        warn "模型未完成下载"
        return 1
    fi
    # 检查基本文件
    if [ ! -f "$MODEL_DIR/tokenizer.json" ] && [ ! -f "$MODEL_DIR/config.json" ]; then
        warn "模型文件缺失 (tokenizer/config)"
        return 1
    fi
    return 0
}

download_from_modelscope() {
    log "📦 从 ModelScope 下载模型: $MODEL"
    mkdir -p "$MODEL_DIR"
    modelscope download --model "$MODEL" --local_dir "$MODEL_DIR" || {
        error "ModelScope 下载失败"; exit 1;
    }
    touch "$MODEL_DIR/.download_complete"
}

download_from_huggingface() {
    log "📦 从 HuggingFace 下载模型: $MODEL"
    mkdir -p "$MODEL_DIR"
    huggingface-cli download "$MODEL" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False || {
        error "HuggingFace 下载失败"; exit 1;
    }
    touch "$MODEL_DIR/.download_complete"
}

convert_model_format() {
    log "🔄 正在转换模型格式..."
    $PYTHON_CMD /app/convert_hf_format.sh "$MODEL_DIR" || warn "转换脚本执行出错 (可能是无影响错误)"
    if [ -d "$MODEL_DIR/vllm" ] && [ -f "$MODEL_DIR/vllm/model.safetensors" ]; then
        touch "$MODEL_DIR/.conversion_complete"
        log "✅ 模型转换完成"
    else
        warn "模型转换结果不完整，请检查 $MODEL_DIR/vllm"
    fi
}

# ---------------- 环境准备 ----------------
log "⚙️ 检查 Python 环境..."
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
    log "✅ 使用 uv 虚拟环境执行 Python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    log "✅ 使用系统 Python3"
else
    error "❌ 未检测到 Python3 或 uv，请检查镜像环境"
    exit 1
fi

# 检查 GPU 是否可见
if command -v nvidia-smi &> /dev/null; then
    log "🧩 GPU 检测成功:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    warn "⚠️ 未检测到 NVIDIA GPU，可能在 CPU 模式下运行"
fi

# ---------------- 依赖安装 ----------------
log "📦 安装依赖..."
$PYTHON_CMD -m pip install --upgrade pip -i "$PYPI_MIRROR"
$PYTHON_CMD -m pip install modelscope huggingface_hub torch gradio vllm -i "$PYPI_MIRROR"

# ---------------- 模型下载与转换 ----------------
if [ "$DOWNLOAD_MODEL" = "1" ]; then
    if ! check_model_exists; then
        if [ "$VLLM_USE_MODELSCOPE" = "1" ]; then
            download_from_modelscope
        else
            download_from_huggingface
        fi
    else
        log "✅ 模型已存在，跳过下载"
    fi
else
    warn "🛑 模型下载已禁用"
fi

if [ "$CONVERT_MODEL" = "1" ]; then
    if [ ! -f "$MODEL_DIR/.conversion_complete" ]; then
        convert_model_format
    else
        log "✅ 模型已转换，跳过"
    fi
fi

# ---------------- 启动服务 ----------------
log "🚀 启动 IndexTTS-2 vLLM API 服务 (端口: $PORT)"
exec $PYTHON_CMD /app/api_server.py \
    --model_dir "$MODEL_DIR" \
    --port "$PORT" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
