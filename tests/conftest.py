"""Pytest configuration and shared fixtures."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI application.

    Yields:
        TestClient instance for making requests.
    """
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock LLM responses for testing without real API calls.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Function to set the mock response.
    """

    def _set_response(response: str):
        def mock_get_response(self, user_message, system_prompt=""):
            return response

        monkeypatch.setattr(
            "src.models.llm.LLMClient.get_response",
            mock_get_response,
        )

    return _set_response
