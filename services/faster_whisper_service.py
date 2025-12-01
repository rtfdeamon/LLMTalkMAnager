"""
Faster Whisper STT service integration for Pipecat.

This module wraps the Faster-Whisper model with buffering logic
based on VAD events so that turn-based speech recognition can run
on local hardware without blocking the event loop.
"""
import asyncio
import io
import logging
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService

logger = logging.getLogger(__name__)


class FasterWhisperSTTService(STTService):
    """Speech-to-text service using Faster-Whisper.

    The service buffers audio while the user is speaking and triggers
    transcription once silence is detected by the VAD component. The
    blocking inference call is executed in a thread pool to avoid
    stalling the asyncio event loop.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        language: str = "ru",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._language = language

        self._model: Optional[WhisperModel] = None
        self._audio_buffer = io.BytesIO()
        self._buffering = False

        self._load_model()

    def _load_model(self) -> None:
        """Load the CTranslate2 Whisper model into memory."""
        logger.info("Loading Faster-Whisper model %s on %s", self._model_size, self._device)
        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("Faster-Whisper model loaded successfully")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load Faster-Whisper model: %s", exc)
            raise

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        logger.debug("FasterWhisperSTTService started")

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        logger.debug("FasterWhisperSTTService stopped")

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        self._audio_buffer = io.BytesIO()
        self._buffering = False
        logger.debug("FasterWhisperSTTService canceled and buffer reset")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking; resetting buffer")
            self._audio_buffer = io.BytesIO()
            self._buffering = True
        elif isinstance(frame, AudioRawFrame) and self._buffering:
            self._audio_buffer.write(frame.audio)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking; starting transcription")
            self._buffering = False
            await self._transcribe_buffered_audio()

    async def _transcribe_buffered_audio(self) -> None:
        audio_data = self._audio_buffer.getvalue()
        if not audio_data:
            logger.warning("Skipping transcription for empty audio buffer")
            return

        self._audio_buffer.seek(0)
        loop = asyncio.get_running_loop()

        try:
            segments, _ = await loop.run_in_executor(
                None, self._run_inference_sync, audio_data
            )
            full_text = " ".join([segment.text for segment in segments]).strip()
            if full_text:
                logger.info("Transcription result: %s", full_text)
                await self.push_frame(
                    TranscriptionFrame(text=full_text, user_id="user", timestamp=0)
                )
            else:
                logger.debug("No speech detected in audio buffer")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Transcription failed: %s", exc)
            await self.push_frame(ErrorFrame(error=str(exc)))
        finally:
            self._audio_buffer = io.BytesIO()

    def _run_inference_sync(self, audio_bytes: bytes) -> Tuple[List, object]:
        audio_int16 = np.frombuffer(audio_bytes, np.int16)
        audio_float32 = audio_int16.flatten().astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio_float32,
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return list(segments), info

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        yield ErrorFrame(error="FasterWhisperSTTService uses buffered VAD mode.")
        # TODO: Support direct streaming mode if Pipecat adds compatible hooks.
