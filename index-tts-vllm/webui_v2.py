import os
import io
import asyncio
import json
import base64
import traceback
import time
import soundfile as sf
from contextlib import asynccontextmanager
from typing import Optional, List, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn
import re
import argparse

from indextts.infer_vllm_v2 import IndexTTS2

logger.add("logs/api_server_stream.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "TTS model not initialized"}
        )
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "message": "Service is running", "timestamp": time.time()}
    )

# ===============================
# 🔹 WebSocket伪流式TTS接口
# ===============================
@app.websocket("/tts_stream")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_json()
        text = data.get("text")
        spk_audio_path = data.get("spk_audio_path")
        emo_control_method = data.get("emo_control_method", 0)
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0]*8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        # 按标点分割文本
        sentences = re.split(r'(?<=[。！？；，,.!?])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        logger.info(f"🔹 接收到文本，共 {len(sentences)} 段: {sentences}")

        # 逐段生成音频并流式发送
        for idx, seg in enumerate(sentences):
            logger.debug(f"正在合成第 {idx+1}/{len(sentences)} 段: {seg}")
            sr, wav = await tts.infer(
                spk_audio_prompt=spk_audio_path,
                text=seg,
                output_path=None,
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=emo_vec if emo_control_method == 2 else None,
                use_emo_text=(emo_control_method==3),
                emo_text=emo_text,
                use_random=emo_random,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence)
            )

            with io.BytesIO() as buf:
                sf.write(buf, wav, sr, format='WAV')
                wav_bytes = buf.getvalue()
            b64_audio = base64.b64encode(wav_bytes).decode('utf-8')

            # 模拟阿里云的WebSocket协议格式
            message = {
                "header": {"status": 1 if idx < len(sentences)-1 else 2},  # 1=中间帧, 2=最后一帧
                "payload": {
                    "index": idx,
                    "audio": b64_audio,
                    "text": seg
                }
            }
            await ws.send_json(message)
            await asyncio.sleep(0.1)  # 模拟流式节奏（可调）

        logger.info("✅ 全部语音段发送完毕。")

    except WebSocketDisconnect:
        logger.warning("❌ 客户端断开连接。")
    except Exception as ex:
        tb = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        logger.error(tb)
        await ws.send_json({"header": {"status": "error"}, "error": str(ex)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM")
    parser.add_argument("--is_fp16", action="store_true", default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)
