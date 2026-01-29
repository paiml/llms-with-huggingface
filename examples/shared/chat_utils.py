"""Common utilities for chat interfaces.

This module provides reusable patterns for interacting with OpenAI-compatible
chat APIs, reducing code duplication across examples.
"""

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion


@dataclass
class ChatConfig:
    """Configuration for chat client.

    Attributes:
        api_key: API key for authentication.
        base_url: Base URL for the API endpoint.
        model_name: Name of the model to use.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in response.
    """

    api_key: str
    base_url: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 800

    @classmethod
    def from_env(cls) -> "ChatConfig":
        """Create configuration from environment variables.

        Returns:
            ChatConfig instance with values from environment.
        """
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
            model_name=os.getenv("MODEL_NAME", "qwen2.5-coder:7b-instruct"),
        )


def create_chat_client(config: ChatConfig) -> OpenAI:
    """Create an OpenAI client from configuration.

    Args:
        config: Chat configuration settings.

    Returns:
        Configured OpenAI client instance.
    """
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def create_message_payload(
    user_message: str,
    system_prompt: str = "You are a friendly AI assistant.",
) -> list[dict[str, Any]]:
    """Create a message payload for chat completion.

    Args:
        user_message: The user's input message.
        system_prompt: System prompt to set assistant behavior.

    Returns:
        List of message dictionaries for the API.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def format_chat_response(completion: ChatCompletion) -> str:
    """Extract text content from chat completion.

    Args:
        completion: ChatCompletion response from the API.

    Returns:
        The text content of the assistant's response.
    """
    if completion.choices and completion.choices[0].message.content:
        return completion.choices[0].message.content
    return ""


def send_chat_message(
    client: OpenAI,
    config: ChatConfig,
    user_message: str,
    system_prompt: str = "You are a friendly AI assistant.",
) -> ChatCompletion:
    """Send a message to the chat API and get a response.

    Args:
        client: OpenAI client instance.
        config: Chat configuration settings.
        user_message: The user's input message.
        system_prompt: System prompt to set assistant behavior.

    Returns:
        ChatCompletion object containing the response.
    """
    messages = create_message_payload(user_message, system_prompt)
    return client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
