"""LLM client for OpenAI-compatible APIs."""

import logging
import os
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client for local and remote models.

    Supports OpenAI-compatible APIs including Ollama, vLLM, and OpenAI.

    Attributes:
        client: OpenAI client instance.
        model: Model name to use for completions.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            api_key: API key for authentication. Defaults to OPENAI_API_KEY env var.
            base_url: Base URL for the API. Defaults to OPENAI_API_BASE env var.
            model: Model name. Defaults to MODEL_NAME env var.

        Raises:
            ValueError: If base_url has invalid schema.
        """
        resolved_base_url = base_url or os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")

        if not resolved_base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url schema: {resolved_base_url}. Must start with http:// or https://")

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=resolved_base_url,
        )
        self.model = model or os.getenv("MODEL_NAME", "qwen2.5-coder:7b-instruct")
        logger.info("LLMClient initialized with model: %s", self.model)

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> ChatCompletion:
        """Send a chat completion request.

        Args:
            messages: List of message dictionaries with role and content.
            temperature: Sampling temperature (0.0-2.0). Must be a float.
            max_tokens: Maximum tokens in response. Enforced by the API.

        Returns:
            ChatCompletion object from the API.

        Raises:
            TypeError: If temperature is not a float.
            ValueError: If temperature is out of range.
        """
        if not isinstance(temperature, float):
            raise TypeError(f"temperature must be float, got {type(temperature)}")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")

        logger.debug("Sending chat request with %d messages", len(messages))
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_response(self, user_message: str, system_prompt: str = "") -> str:
        """Get a simple text response.

        Args:
            user_message: The user's input message.
            system_prompt: Optional system prompt to set behavior.

        Returns:
            The text content of the assistant's response.
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        completion = self.chat(messages)
        return completion.choices[0].message.content or ""
