import os
import io
import json
import time
import traceback
import soundfile as sf
import asyncio
import argparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import uvicorn

from indextts.infer_vllm_v2 import IndexTTS2


logger.add("logs/api_server_v2_ws.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

tts = None
args = None  # 用于 lifespan 内部访问命令行参数


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动与关闭生命周期"""
    global tts
    logger.info("🔄 正在初始化 IndexTTS2 模型...")
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    logger.info("✅ 模型加载完成.")
    yield
    logger.info("🧹 清理资源完成.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 测试环境开放，生产请指定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "timestamp": time.time()})

@app.websocket("/ws/test")
async def websocket_tts_test(websocket: WebSocket):
    logger.info(f"收到WebSocket连接请求 from: {websocket.client}")
    
    # 记录查询参数
    query_params = dict(websocket.query_params)
    logger.info(f"查询参数: {query_params}")
    
    token = websocket.query_params.get("token", "")
    logger.info(f"提取到的token: '{token}'")
    logger.info(f"期望的token: 'my_test_token_123'")
    logger.info(f"Token匹配结果: {token == 'my_test_token_123'}")
    
    if token != "my_test_token_123":
        logger.warning(f"Token验证失败，关闭连接，返回code 403")
        await websocket.close(code=403)
        return

    logger.info("Token验证成功，接受连接")
    await websocket.accept()
    await websocket.send_text("欢迎连接到 TTS WebSocket 服务!")
    logger.info("已发送欢迎消息")

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"收到客户端消息: {data}")
            # 返回模拟的音频数据
            await websocket.send_bytes(b"FAKE_AUDIO_BYTES")
            logger.info("已发送模拟音频数据")
    except Exception as e:
        logger.error(f"WebSocket处理异常: {e}")
        import traceback
        logger.error(f"异常详情: {traceback.format_exc()}")
    finally:
        logger.info("WebSocket连接关闭")


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket, token: str = Query(None)):
    """基于FastAPI WebSocket实现的 TTS WebSocket 接口"""
    # token 验证
    if token != "my_test_token_123":
        await websocket.close(code=403)
        logger.warning(f"❌ 拒绝连接，非法 token: {token}")
        return

    await websocket.accept()
    logger.info(f"✅ WebSocket 已连接: {websocket.client.host}")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # 客户端要求关闭连接
            if message.get("header", {}).get("status") == 2:
                logger.info("🧾 客户端请求关闭连接。")
                await websocket.close()
                break

            text = message.get("payload", {}).get("text", "")
            spk_audio_path = message.get("payload", {}).get("spk_audio_path", "")

            if not text or not spk_audio_path:
                await websocket.send_json({
                    "header": {"code": 400, "message": "缺少 text 或 spk_audio_path 参数"}
                })
                continue

            # TTS 推理
            sr, wav = await tts.infer(
                spk_audio_prompt=spk_audio_path,
                text=text,
                output_path=None,
                emo_audio_prompt=None,
                emo_alpha=1.0,
                emo_vector=None,
            )

            # 转WAV字节流
            with io.BytesIO() as buf:
                sf.write(buf, wav, sr, format='WAV')
                audio_bytes = buf.getvalue()

            await websocket.send_bytes(audio_bytes)

    except WebSocketDisconnect:
        logger.info("🔌 客户端断开连接。")
    except Exception as e:
        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"⚠️ 异常: {tb}")
        await websocket.send_json({"header": {"code": 500, "message": str(e)}})
        await websocket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7800)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app, host=args.host, port=args.port)
