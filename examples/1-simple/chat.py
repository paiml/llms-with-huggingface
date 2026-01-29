"""Simple chat interface using OpenAI-compatible API."""

import os
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

try:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
client.api_base = "http://localhost:11434"
model_name: str | None = os.getenv("MODEL_NAME")


def ai_chat(user_message: str) -> ChatCompletion:
    """Send a message to the AI and get a response.

    Args:
        user_message: The user's input message.

    Returns:
        ChatCompletion object containing the AI's response.
    """
    message_text: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant that helps people find information and answer questions.",
        },
        {"role": "user", "content": user_message},
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    return completion


def print_response(text: str) -> None:
    """Print response with rich formatting if available."""
    if RICH_AVAILABLE:
        console.print(Panel(text, title="AI Response", border_style="green"))
    else:
        print(text)


if RICH_AVAILABLE:
    console.print("[bold cyan]Welcome![/bold cyan] How can I help you today?")
else:
    print("Welcome! How can I help you today?")

while True:
    user_message = input(">> ")
    completion = ai_chat(user_message)
    response = completion.choices[0].message.content
    print_response(response)
