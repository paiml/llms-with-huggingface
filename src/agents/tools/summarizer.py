"""Text summarization tool for the research agent."""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Maximum input length to prevent memory issues
MAX_INPUT_LENGTH = 50000


@tool
def summarize_text(text: str, max_length: int = 200) -> str:
    """Summarize a long piece of text into key points.

    Use this tool to condense long documents, articles, or search results
    into a concise summary.

    Args:
        text: The text to summarize. Will be truncated if too long.
        max_length: Target maximum length of the summary in words.

    Returns:
        A concise summary of the text, or error message on failure.
    """
    if not text or not text.strip():
        return "Error: Empty text provided for summarization"

    # Truncate very long input to prevent memory issues
    if len(text) > MAX_INPUT_LENGTH:
        logger.warning(
            "Truncating summarization input from %d to %d chars",
            len(text),
            MAX_INPUT_LENGTH,
        )
        text = text[:MAX_INPUT_LENGTH] + "... [truncated]"

    # Validate max_length parameter
    if not isinstance(max_length, int) or max_length < 1:
        max_length = 200
        logger.warning("Invalid max_length, using default: %d", max_length)

    logger.info("Summarizing %d chars to ~%d words", len(text), max_length)

    try:
        # Import here to avoid circular dependency
        from src.models.llm import LLMClient

        client = LLMClient()
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
        return client.get_response(
            prompt,
            system_prompt="You are a concise summarizer. Extract key points only.",
        )
    except Exception as e:
        logger.error("Summarization failed: %s", e)
        return f"Error: Summarization failed - {e}"
