"""Web search tool for the research agent."""

import logging

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Connection pool for HTTP requests
_http_client: httpx.Client | None = None


def get_http_client() -> httpx.Client:
    """Get or create a pooled HTTP client.

    Returns:
        httpx.Client with connection pooling enabled.
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=30.0)
    return _http_client


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic.

    Use this tool to find recent news, current events, or information
    not available in the knowledge base.

    Args:
        query: The search query to look up. Should be specific and concise.

    Returns:
        Search results as a formatted string, or error message on failure.
    """
    if not query or not query.strip():
        return "Error: Empty search query provided"

    logger.info("Web search for: %s", query[:100])

    # Mock implementation - in production, use a real search API
    # (Google Custom Search, Bing Search API, SerpAPI, etc.)
    try:
        # Simulate a search result
        return f"[Mock Web Search Results for '{query}']\n\n" + (
            "1. Wikipedia: Overview and background information\n"
            "2. News Article: Recent developments and updates\n"
            "3. Research Paper: Academic perspective on the topic\n\n"
            "Note: This is a mock response. Configure a real search API for production."
        )
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return f"Error: Web search failed - {e}"
