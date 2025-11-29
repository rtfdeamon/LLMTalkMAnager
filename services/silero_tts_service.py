"""
Silero-based TTS service for Pipecat with sentence-level synthesis.
"""
import asyncio
import logging
from typing import AsyncGenerator

import numpy as np
import torch

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.tts_service import TTSService

logger = logging.getLogger(__name__)


class SileroTTSService(TTSService):
    """Text-to-speech service powered by Silero models."""

    def __init__(
        self,
        language: str = "ru",
        speaker: str = "xenia",
        sample_rate: int = 24000,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._language = language
        self._speaker = speaker
        self._sample_rate = sample_rate
        self._device = torch.device(device)
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Loading Silero TTS model: language=%s speaker=%s", self._language, self._speaker)
        try:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=self._language,
                speaker="v5_ru",
            )
            self._model.to(self._device)
            logger.info("Silero TTS model loaded successfully")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load Silero TTS model: %s", exc)
            raise

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            logger.debug("Skipping TTS for empty text input")
            return

        yield TTSStartedFrame()
        loop = asyncio.get_running_loop()
        try:
            audio_tensor = await loop.run_in_executor(None, self._generate_audio_sync, text)
            audio_int16 = (audio_tensor * 32767).clamp(-32768, 32767).to(torch.int16)
            audio_bytes = audio_int16.cpu().numpy().tobytes()
            yield TTSAudioRawFrame(audio=audio_bytes, sample_rate=self._sample_rate, num_channels=1)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("TTS generation failed: %s", exc)
            yield ErrorFrame(error=str(exc))
        finally:
            yield TTSStoppedFrame()

    def _generate_audio_sync(self, text: str):
        return self._model.apply_tts(
            text=text,
            speaker=self._speaker,
            sample_rate=self._sample_rate,
            put_accent=True,
            put_yo=True,
        )
        # TODO: expose speaking speed and voice effects as runtime parameters.
