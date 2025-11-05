import asyncio
import websockets
import logging
import traceback

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 方法1：适配新版本签名
async def handle_connection(websocket):
    print(f"客户端连接成功: {websocket.remote_address}")
    
    try:
        # 先发送一个欢迎消息测试连接
        welcome_msg = "欢迎连接到WebSocket服务器"
        await websocket.send(welcome_msg)
        print(f"已发送欢迎消息: {welcome_msg}")
        
        async for message in websocket:
            print(f"收到消息: {message}")
            
            # 简单的响应处理
            response = f"服务器收到了: {message}"
            print(f"准备发送回复: {response}")
            
            try:
                await websocket.send(response)
                print("回复发送成功")
            except Exception as send_error:
                print(f"发送回复时出错: {send_error}")
                break
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"客户端断开连接: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"处理连接时发生错误: {e}")
        traceback.print_exc()
    finally:
        print("连接处理结束")
async def main():
    # 使用旧的创建方式确保兼容性
    async with websockets.serve(
        handle_connection,
        "localhost", 
        7800,
        origins=None
    ):  
        print("WebSocket 服务器启动在 ws://localhost:7800")
        await asyncio.Future()  # 永久运行

if __name__ == "__main__":
    asyncio.run(main())