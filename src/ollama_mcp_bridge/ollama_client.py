"""Thin wrapper around the official ollama Python package.

WHY A WRAPPER (vs using ollama.AsyncClient directly):
    - Error mapping: ollama raises generic ResponseError; we map to typed exceptions
      (OllamaConnectionError, OllamaModelError, OllamaResponseError) so the loop
      and bridge can handle each case differently.
    - Tool call extraction: model responses contain tool_calls in a nested structure;
      extract_tool_calls() converts them to our OllamaToolCall type for uniform handling.
    - Health check: check_health() and list_models() for bridge startup validation.

CRITICAL: The ollama package uses the native /api/chat endpoint by default.
DO NOT switch to /v1/chat/completions — that endpoint silently drops tool_calls
when streaming (confirmed Ollama bug, github.com/ollama/ollama/issues/12557).
The native endpoint handles streaming + tool calls correctly.

This module is a pure transport layer — no security logic. Every response from
this client is UNTRUSTED. Tool calls in the response are model-generated and
must be validated by SecurityGateway before execution.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ollama import AsyncClient, ChatResponse, ResponseError

from .errors import OllamaConnectionError, OllamaModelError, OllamaResponseError
from .types import OllamaToolCall

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async Ollama API client wrapper.

    Wraps ollama.AsyncClient with error mapping and health checks.
    """

    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self._host = host
        self._client = AsyncClient(host=host)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> ChatResponse:
        """Send a chat request to Ollama.

        Args:
            model: Model name (e.g., 'llama3.1:8b').
            messages: Conversation history.
            tools: Ollama-format tool definitions.
            stream: Whether to stream the response.

        Returns:
            ChatResponse with .message.content and .message.tool_calls.

        Raises:
            OllamaConnectionError: Cannot reach Ollama.
            OllamaModelError: Model not found or unavailable.
            OllamaResponseError: Malformed response.
        """
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            # Non-streaming for tool calls — simpler and more reliable.
            # Streaming support via chat_stream() below.
            response = await self._client.chat(**kwargs)
            return response
        except ResponseError as e:
            if "not found" in str(e).lower() or "pull" in str(e).lower():
                raise OllamaModelError(f"Model '{model}' not available: {e}") from e
            raise OllamaResponseError(f"Ollama API error: {e}") from e
        except Exception as e:
            if "connect" in str(e).lower() or "refused" in str(e).lower():
                raise OllamaConnectionError(
                    f"Cannot reach Ollama at {self._host}: {e}"
                ) from e
            raise OllamaResponseError(f"Unexpected Ollama error: {e}") from e

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatResponse]:
        """Stream a chat response from Ollama.

        Yields ChatResponse chunks. Text tokens arrive incrementally.
        Tool calls arrive in the final chunk(s).
        """
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
            if tools:
                kwargs["tools"] = tools
            async for chunk in await self._client.chat(**kwargs):
                yield chunk
        except ResponseError as e:
            if "not found" in str(e).lower():
                raise OllamaModelError(f"Model '{model}' not available: {e}") from e
            raise OllamaResponseError(f"Ollama streaming error: {e}") from e
        except Exception as e:
            if "connect" in str(e).lower() or "refused" in str(e).lower():
                raise OllamaConnectionError(
                    f"Cannot reach Ollama at {self._host}: {e}"
                ) from e
            raise OllamaResponseError(f"Unexpected Ollama streaming error: {e}") from e

    async def check_health(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            await self._client.list()
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available model names."""
        try:
            result = await self._client.list()
            return [m.model for m in result.models]
        except Exception as e:
            raise OllamaConnectionError(f"Cannot list models: {e}") from e

    @staticmethod
    def extract_tool_calls(response: ChatResponse) -> list[OllamaToolCall]:
        """Extract tool calls from an Ollama chat response.

        Returns empty list if no tool calls in response.
        """
        if not response.message or not response.message.tool_calls:
            return []

        calls = []
        for tc in response.message.tool_calls:
            calls.append(
                OllamaToolCall(
                    function_name=tc.function.name,
                    arguments=tc.function.arguments or {},
                )
            )
        return calls
