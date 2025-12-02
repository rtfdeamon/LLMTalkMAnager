import torch
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugSilero")

def load_silero():
    logger.info("Starting Silero load...")
    try:
        device = torch.device("cpu")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v5_ru",
            trust_repo=True,
            verbose=True
        )
        model.to(device)
        logger.info("Silero loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_silero()
