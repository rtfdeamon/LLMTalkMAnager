"""
Entry point that wires together local STT/LLM/TTS services for Pipecat.
"""
import asyncio
import logging
import os

from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.transports.services.daily import DailyParams, DailyTransport

from services.faster_whisper_service import FasterWhisperSTTService
from services.ollama_service import CustomOllamaLLMService
from services.silero_tts_service import SileroTTSService

logger = logging.getLogger("LocalVoiceBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _build_transport() -> DailyTransport:
    load_dotenv()
    try:
        transport = DailyTransport(
            room_url=os.getenv("DAILY_ROOM_URL"),
            token=os.getenv("DAILY_TOKEN"),
            bot_name="Локальный Ассистент",
            params=DailyParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_audio_passthrough=True,
            ),
        )
        logger.info("Daily transport initialized")
        return transport
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to initialize Daily transport: %s", exc)
        raise


def _build_pipeline() -> Pipeline:
    try:
        pipeline = Pipeline()
        SentenceAggregator(pipeline)
        logger.info("Pipeline with SentenceAggregator created")
        return pipeline
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to create pipeline: %s", exc)
        raise


def _build_services() -> tuple[SileroVADAnalyzer, FasterWhisperSTTService, CustomOllamaLLMService, SileroTTSService]:
    try:
        vad_analyzer = SileroVADAnalyzer()
        stt = FasterWhisperSTTService(model_size="small", device="cuda", language="ru")
        llm = CustomOllamaLLMService(model="llama3", base_url="http://localhost:11434/v1", num_ctx=4096)
        tts = SileroTTSService(language="ru", speaker="xenia", sample_rate=24000, device="cpu")
        logger.info("Services initialized successfully")
        return vad_analyzer, stt, llm, tts
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to initialize services: %s", exc)
        raise


async def _run_task(runner: PipelineRunner, task: PipelineTask) -> None:
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info("Pipeline task cancelled")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Pipeline task failed: %s", exc)
        raise
    finally:
        logger.info("Pipeline task completed")


async def main() -> None:
    transport = _build_transport()
    pipeline = _build_pipeline()
    vad_analyzer, stt, llm, tts = _build_services()

    runner = PipelineRunner()
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, enable_metrics=True),
        transport=transport,
    )

    await _run_task(runner, task)


if __name__ == "__main__":
    asyncio.run(main())
    # TODO: register graceful signal handling for long-running deployments.
