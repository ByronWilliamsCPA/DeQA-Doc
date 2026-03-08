"""VLM API clients for IQA teacher evaluation.

Provides a unified protocol and two concrete implementations:
- AnthropicClient: Direct Anthropic API via anthropic SDK
- OpenRouterClient: OpenRouter API via openai SDK
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class VLMResponse:
    """Raw response from a VLM API call."""

    text: str
    latency_ms: int
    model: str
    provider: str


class VLMClient(Protocol):
    """Protocol for VLM API clients."""

    def rate_image(
        self,
        image_b64: str,
        media_type: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> VLMResponse:
        """Send an image to the VLM and get a raw text response."""
        ...


class AnthropicClient:
    """Direct Anthropic API client.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model: Model identifier.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def rate_image(
        self,
        image_b64: str,
        media_type: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> VLMResponse:
        """Rate an image via the Anthropic Messages API."""
        start = time.time()

        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )

        latency_ms = int((time.time() - start) * 1000)
        text = response.content[0].text
        return VLMResponse(
            text=text,
            latency_ms=latency_ms,
            model=self._model,
            provider="anthropic",
        )


class OpenRouterClient:
    """OpenRouter API client using the OpenAI SDK.

    Args:
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
        model: Model identifier on OpenRouter.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "anthropic/claude-sonnet-4-6",
    ) -> None:
        import os

        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client = OpenAI(
            api_key=resolved_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._model = model

    def rate_image(
        self,
        image_b64: str,
        media_type: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> VLMResponse:
        """Rate an image via the OpenRouter API (OpenAI-compatible)."""
        start = time.time()

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        )

        latency_ms = int((time.time() - start) * 1000)
        text: Any = response.choices[0].message.content or ""
        return VLMResponse(
            text=str(text),
            latency_ms=latency_ms,
            model=self._model,
            provider="openrouter",
        )
