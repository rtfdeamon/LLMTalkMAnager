import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImportVerifier")

def check_imports():
    try:
        logger.info("Checking imports...")
        import pipecat
        import faster_whisper
        import torch
        import torchaudio
        import dotenv
        import aiohttp
        import numpy
        
        # Check local services
        from services.faster_whisper_service import FasterWhisperSTTService
        from services.ollama_service import CustomOllamaLLMService
        from services.silero_tts_service import SileroTTSService
        
        logger.info("All imports successful!")
        return True
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if check_imports():
        sys.exit(0)
    else:
        sys.exit(1)
