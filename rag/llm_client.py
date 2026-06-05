"""
rag/llm_client.py
Cliente HTTP para o Ollama com retry, timeout e fallback de modelo.
"""

from __future__ import annotations

import structlog
import time
from dataclasses import dataclass

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings
from domain.exceptions import LLMTimeoutError, LLMUnavailableError

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    fallback_used: bool = False


class OllamaClient:
    """
    Cliente para o servidor Ollama local.

    Fluxo:
    1. Tenta modelo primário (Mistral 7B) com retry exponencial
    2. Se falhar após N tentativas, tenta modelo fallback (Qwen 2.5 3B)
    3. Se ambos falharem, lança LLMUnavailableError
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._generate_url = settings.ollama_generate_url
        self._primary_model = settings.llm_primary_model
        self._fallback_model = settings.llm_fallback_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries

    def generate(
        self,
        system_prompt: str,
        user_message: str,
    ) -> LLMResponse:
        """Gera resposta com retry no modelo primário e fallback automático."""
        start = time.monotonic()

        # Tenta modelo primário
        try:
            text, tokens_in, tokens_out = self._call_with_retry(
                model=self._primary_model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
            latency = int((time.monotonic() - start) * 1000)
            return LLMResponse(
                text=text,
                model_used=self._primary_model,
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                latency_ms=latency,
                fallback_used=False,
            )

        except (LLMTimeoutError, LLMUnavailableError) as primary_exc:
            logger.warning(
                "Primary model failed, trying fallback",
                primary=self._primary_model,
                fallback=self._fallback_model,
                error=str(primary_exc),
            )

        # Tenta modelo fallback
        try:
            text, tokens_in, tokens_out = self._call_with_retry(
                model=self._fallback_model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
            latency = int((time.monotonic() - start) * 1000)
            logger.info("Fallback model used", extra={"model": self._fallback_model})
            return LLMResponse(
                text=text,
                model_used=self._fallback_model,
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                latency_ms=latency,
                fallback_used=True,
            )

        except Exception as fallback_exc:
            raise LLMUnavailableError(
                f"Both primary and fallback models failed. "
                f"Primary: {self._primary_model}, Fallback: {self._fallback_model}. "
                f"Last error: {fallback_exc}"
            ) from fallback_exc

    def _call_with_retry(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, int, int]:
        """Chama Ollama com retry exponencial."""

        @retry(
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, 30),  # 30 = logging.WARNING
            reraise=False,
        )
        def _inner() -> tuple[str, int, int]:
            return self._call_ollama(model, system_prompt, user_message)

        try:
            return _inner()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama timeout after {self._timeout}s") from exc
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"Cannot connect to Ollama: {exc}") from exc

    def _call_ollama(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, int, int]:
        """Realiza chamada HTTP ao Ollama e retorna (texto, tokens_in, tokens_out)."""
        payload = {
            "model": model,
            "prompt": user_message,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
                "stop": [],
            },
        }

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._generate_url, json=payload)
            response.raise_for_status()

        data = response.json()
        text: str = data.get("response", "").strip()
        tokens_in: int = data.get("prompt_eval_count", 0)
        tokens_out: int = data.get("eval_count", 0)

        if not text:
            raise LLMUnavailableError("Empty response from Ollama")

        logger.debug(
            "LLM response received",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            response_len=len(text),
        )

        return text, tokens_in, tokens_out

    def check_availability(self) -> bool:
        """Verifica se o servidor Ollama está acessível."""
        settings = get_settings()
        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(settings.ollama_tags_url)
                return r.status_code == 200
        except Exception:
            return False
