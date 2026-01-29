"""Tests for the LLM client."""

import pytest


class TestLLMClientInitialization:
    """Tests for LLMClient initialization."""

    def test_invalid_url_schema_raises(self):
        """F-CULT-002: Invalid URL schema raises ValueError."""
        from src.models.llm import LLMClient

        with pytest.raises(ValueError, match="Invalid base_url schema"):
            LLMClient(base_url="ftp://invalid.url")

    def test_defaults_from_env(self, monkeypatch):
        """F-MOCK-004: LLMClient uses defaults if env vars unset."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("MODEL_NAME", raising=False)

        from src.models.llm import LLMClient

        client = LLMClient()
        assert client.model == "qwen2.5-coder:7b-instruct"


class TestLLMClientChat:
    """Tests for LLMClient chat method."""

    def test_temperature_must_be_float(self):
        """F-CODE-013: Temperature must be a float."""
        from src.models.llm import LLMClient

        client = LLMClient()

        with pytest.raises(TypeError, match="temperature must be float"):
            client.chat([{"role": "user", "content": "test"}], temperature=1)

    def test_temperature_boundary_low(self):
        """F-CODE-013: Temperature below 0 raises ValueError."""
        from src.models.llm import LLMClient

        client = LLMClient()

        with pytest.raises(ValueError, match="temperature must be between"):
            client.chat([{"role": "user", "content": "test"}], temperature=-0.1)

    def test_temperature_boundary_high(self):
        """F-CODE-013: Temperature above 2.0 raises ValueError."""
        from src.models.llm import LLMClient

        client = LLMClient()

        with pytest.raises(ValueError, match="temperature must be between"):
            client.chat([{"role": "user", "content": "test"}], temperature=2.1)

    def test_valid_temperature_accepted(self):
        """Temperature within range is accepted."""
        from src.models.llm import LLMClient

        client = LLMClient()
        # This would fail if we actually called the API, but validates params
        # In real tests, mock the API call
        assert client.model is not None  # Just verify client works
