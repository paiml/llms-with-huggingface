"""Chat endpoint routes."""

import logging

from fastapi import APIRouter, HTTPException

from src.models.llm import LLMClient
from src.models.schemas import ChatRequest, ChatResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Lazy initialization of LLM client
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client singleton.

    Returns:
        LLMClient instance.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def chat(request: ChatRequest) -> ChatResponse:
    """Simple chat endpoint for LLM interactions.

    Send a message and receive a response from the language model.

    Args:
        request: Chat request with message and optional system prompt.

    Returns:
        Chat response with the assistant's reply.

    Raises:
        HTTPException: On LLM errors (500) or validation errors (422).
    """
    logger.info("Chat request: %s...", request.message[:50])

    try:
        client = get_llm_client()
        response = client.get_response(
            request.message,
            system_prompt=request.system_prompt,
        )
        return ChatResponse(response=response)
    except Exception as e:
        logger.error("Chat failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"detail": str(e), "error_code": "LLM_ERROR"},
        )
