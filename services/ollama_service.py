"""
Custom LLM service for Pipecat that targets the local Ollama API.
"""
import logging
from typing import Dict, Any

from pipecat.services.openai.llm import OpenAILLMService

logger = logging.getLogger(__name__)


class CustomOllamaLLMService(OpenAILLMService):
    """OpenAI-compatible LLM service configured for Ollama."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        keep_alive: str = "-1m",
        num_ctx: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, base_url=base_url, api_key=api_key, **kwargs)
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx
        self._temperature = temperature
        logger.info(
            "Initialized CustomOllamaLLMService with model=%s, base_url=%s", model, base_url
        )

    def _get_extra_body(self) -> Dict[str, Any]:
        try:
            return {
                "keep_alive": self._keep_alive,
                "options": {
                    "num_ctx": self._num_ctx,
                    "temperature": self._temperature,
                    "top_k": 40,
                    "top_p": 0.9,
                },
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to build extra body for Ollama request: %s", exc)
            # TODO: investigate structured error propagation to caller.
            raise
