import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.ollama_service import CustomOllamaLLMService
from services.silero_tts_service import SileroTTSService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocalVoiceBot")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global services
_llm = None
_tts = None


@app.on_event("startup")
async def startup_event():
    """Preload AI models at server startup"""
    global _llm, _tts
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING SERVER - TEXT CHAT MODE")
    logger.info("=" * 60)
    
    logger.info("📥 Loading LLM (Qwen2.5 7B)...")
    _llm = CustomOllamaLLMService(model="qwen2.5:7b", base_url="http://localhost:11434/v1", num_ctx=4096)
    logger.info("✅ LLM Ready (Qwen2.5)")
    
    logger.info("📥 TTS switched to Edge-TTS (Natural Voice)...")
    # _tts = SileroTTSService(language="ru", speaker="xenia", sample_rate=24000, device="cpu")
    logger.info("✅ TTS Ready (Edge-TTS)")
    
    logger.info("=" * 60)
    logger.info("🎉 SERVER READY - Type to chat, voice coming soon!")
    logger.info("=" * 60)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ Client connected!")
    
    # Send component statuses
    await websocket.send_text("status:✅ STT: Browser (Ready)")
    await websocket.send_text("status:✅ LLM: Ready")
    await websocket.send_text("status:✅ TTS: Ready (Edge-TTS)")
    
    await websocket.send_text("status:✅ Ready! Type a message to chat.")
    await websocket.send_text("progress:100")
    
    try:
        while True:
            # Receive text message from client
            message = await websocket.receive_text()
            
            if message.startswith("text:"):
                user_text = message[5:]
                logger.info(f"📝 User: {user_text}")
                
                # Streaming LLM + Sentence-level TTS
                import httpx
                import io
                import re
                import os
                import edge_tts
                
                # Helper for Edge-TTS (Async)
                async def generate_audio_edge(text):
                    try:
                        # Use a high-quality Russian voice
                        voice = "ru-RU-DmitryNeural" # or ru-RU-SvetlanaNeural
                        communicate = edge_tts.Communicate(text, voice)
                        
                        # Save to temp file (Edge-TTS writes to file)
                        temp_file = f"temp_{os.getpid()}_{id(text)}.mp3"
                        await communicate.save(temp_file)
                        
                        if os.path.exists(temp_file):
                            with open(temp_file, "rb") as f:
                                audio_bytes = f.read()
                            os.remove(temp_file)
                            return audio_bytes
                    except Exception as e:
                        logger.error(f"Edge-TTS Error: {e}")
                    return None

                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        "http://localhost:11434/v1/chat/completions",
                        json={
                            "model": "qwen2.5:7b",
                            "messages": [
                                {"role": "system", "content": "Ты - голосовой ассистент. Твоя главная задача - говорить ИСКЛЮЧИТЕЛЬНО на русском языке. Даже если тебя спрашивают на английском или другом языке, ты ОБЯЗАН отвечать только на русском. Никогда не используй английские слова. Отвечай кратко, емко и дружелюбно."},
                                {"role": "user", "content": user_text}
                            ],
                            "stream": True
                        },
                        timeout=120.0
                    ) as response:
                        buffer = ""
                        full_response = ""
                        
                        async for line in response.aiter_lines():
                            if not line or line == "data: [DONE]":
                                continue
                            
                            if line.startswith("data: "):
                                import json
                                try:
                                    data = json.loads(line[6:])
                                    if "content" in data["choices"][0]["delta"]:
                                        token = data["choices"][0]["delta"]["content"]
                                        buffer += token
                                        full_response += token
                                        
                                        # Check for sentence delimiters
                                        if re.search(r'[.!?\n]', token):
                                            # Find the last sentence end
                                            match = re.search(r'(.*[.!?\n])', buffer)
                                            if match:
                                                sentence = match.group(1).strip()
                                                if sentence:
                                                    logger.info(f"🗣️ Speaking: {sentence}")
                                                    await websocket.send_text(f"bot:{sentence}")
                                                    
                                                    # Generate TTS for this sentence (Edge-TTS)
                                                    audio_bytes = await generate_audio_edge(sentence)
                                                    if audio_bytes:
                                                        await websocket.send_bytes(audio_bytes)
                                                
                                                # Keep the rest of the buffer
                                                buffer = buffer[len(match.group(1)):]
                                except Exception as e:
                                    logger.error(f"Error parsing stream: {e}")
                        
                        # Process remaining buffer
                        if buffer.strip():
                            logger.info(f"🗣️ Speaking (final): {buffer}")
                            await websocket.send_text(f"bot:{buffer}")
                            
                            audio_bytes = await generate_audio_edge(buffer)
                            if audio_bytes:
                                await websocket.send_bytes(audio_bytes)
                            
                        logger.info(f"🤖 Full response: {full_response}")
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Connection closed")
