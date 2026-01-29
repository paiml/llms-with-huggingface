"""Mock verification tests - Section VI Falsification."""


class TestMockVerification:
    """Tests to verify mocks don't hide real failures."""

    def test_llm_client_defaults_without_env(self, monkeypatch):
        """F-MOCK-004: LLMClient uses defaults when env vars unset."""
        # Clear environment variables
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("MODEL_NAME", raising=False)

        from src.models.llm import LLMClient

        # Should not crash, should use defaults
        client = LLMClient()
        assert client.model == "qwen2.5-coder:7b-instruct"

    def test_vectorstore_works_without_network(self):
        """F-MOCK-002: VectorStore works without network (in-memory)."""
        from src.rag.vectorstore import VectorStore

        # Should work completely offline
        store = VectorStore(collection_name="offline_test")
        store.add_documents([{"content": "Test document"}])
        results = store.search("test")

        assert len(results) == 1

    def test_web_search_mock_returns_valid_response(self):
        """F-MOCK-003: Web search mock returns valid response."""
        from src.agents.tools.web_search import web_search

        result = web_search.invoke("test query")

        # Mock should return a string, not raise
        assert isinstance(result, str)
        assert len(result) > 0


class TestEnvironmentConfiguration:
    """Tests for environment configuration."""

    def test_respects_custom_base_url(self, monkeypatch):
        """F-ENV-004: LLMClient respects OPENAI_API_BASE."""
        monkeypatch.setenv("OPENAI_API_BASE", "http://custom.server:8080/v1")

        from src.models.llm import LLMClient

        client = LLMClient()
        assert "custom.server" in str(client.client.base_url)

    def test_respects_model_name(self, monkeypatch):
        """F-ENV-005: LLMClient respects MODEL_NAME."""
        monkeypatch.setenv("MODEL_NAME", "custom-model-name")

        from src.models.llm import LLMClient

        client = LLMClient()
        assert client.model == "custom-model-name"
