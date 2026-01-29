"""FastAPI chat endpoint for small language models."""

import os
from typing import Any

import openai
from fastapi import FastAPI
from pydantic import BaseModel


openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
model_name: str | None = os.getenv("MODEL_NAME")


app = FastAPI()


class Body(BaseModel):
    """Request body for the chat endpoint."""

    text: str


def ai_chat(user_message: str) -> dict[str, Any]:
    """Send a message to the small language model and get a response.

    Args:
        user_message: The user's input message.

    Returns:
        Dictionary containing the completion response.
    """
    message_text: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant that helps people find information and answer questions.",
        },
        {"role": "user", "content": user_message},
    ]

    completion = openai.ChatCompletion.create(
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


@app.post("/generate")
def generate(body: Body) -> dict[str, str]:
    """Generate a chat response from user input.

    Args:
        body: Request body containing the user's text.

    Returns:
        Dictionary with the AI's response text.
    """
    completion = ai_chat(body.text)
    return {"text": completion["choices"][0]["message"]["content"]}
