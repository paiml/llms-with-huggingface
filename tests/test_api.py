"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """F-API-001: GET /health returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient):
        """F-API-001: Health response contains status healthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestChatEndpoint:
    """Tests for the /chat endpoint."""

    def test_chat_missing_message_returns_422(self, client: TestClient):
        """F-API-003: POST /chat with missing message field returns 422."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_empty_message_returns_422(self, client: TestClient):
        """F-API-003: POST /chat with empty message returns 422."""
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422

    def test_chat_extra_fields_ignored(self, client: TestClient, mock_llm_response):
        """F-API-004: POST /chat with extra fields ignores them."""
        mock_llm_response("Test response")
        response = client.post(
            "/chat",
            json={"message": "Hello", "extra_field": "ignored"},
        )
        # Should succeed (extra fields ignored by Pydantic)
        assert response.status_code == 200

    def test_chat_valid_request(self, client: TestClient, mock_llm_response):
        """F-API-002: POST /chat with valid JSON returns 200."""
        mock_llm_response("Hello! How can I help you?")
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 200
        assert "response" in response.json()


class TestResearchEndpoint:
    """Tests for the /research endpoint."""

    def test_research_missing_query_returns_422(self, client: TestClient):
        """F-API-005: POST /research requires query field."""
        response = client.post("/research", json={})
        assert response.status_code == 422


class TestDocumentsEndpoint:
    """Tests for the /documents endpoint."""

    def test_documents_missing_fields_returns_422(self, client: TestClient):
        """F-API-007: POST /documents requires title and content."""
        response = client.post("/documents", json={})
        assert response.status_code == 422

    def test_documents_empty_title_returns_422(self, client: TestClient):
        """POST /documents with empty title returns 422."""
        response = client.post(
            "/documents",
            json={"title": "", "content": "Some content"},
        )
        assert response.status_code == 422


class TestMalformedRequests:
    """Tests for malformed request handling."""

    def test_malformed_json_returns_422(self, client: TestClient):
        """F-API-010: Malformed JSON returns error."""
        response = client.post(
            "/chat",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestDocumentation:
    """Tests for API documentation endpoints."""

    def test_swagger_ui_loads(self, client: TestClient):
        """F-API-013: Swagger UI (/docs) loads."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_loads(self, client: TestClient):
        """F-API-014: ReDoc (/redoc) loads."""
        response = client.get("/redoc")
        assert response.status_code == 200
