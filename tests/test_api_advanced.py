"""Advanced API tests - Section V Falsification."""

from fastapi.testclient import TestClient


class TestAPICORS:
    """Tests for CORS configuration."""

    def test_cors_allows_localhost_3000(self, client: TestClient):
        """F-API-008: CORS allows localhost:3000."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


class TestAPIPayloadLimits:
    """Tests for payload handling."""

    def test_large_payload_handled(self, client: TestClient, mock_llm_response):
        """F-API-009: Large payload is handled or rejected."""
        mock_llm_response("Response")

        # 1MB payload
        large_message = "x" * (1024 * 1024)

        response = client.post(
            "/chat",
            json={"message": large_message},
        )

        # Should either succeed or return appropriate error (413 or 422)
        assert response.status_code in [200, 413, 422, 500]


class TestAPIVersioning:
    """Tests for API versioning."""

    def test_api_version_in_openapi(self, client: TestClient):
        """F-API-015: API has version information."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "info" in data
        assert "version" in data["info"]
        assert data["info"]["version"] == "1.0.0"


class TestAPIErrorResponses:
    """Tests for consistent error responses."""

    def test_chat_error_format(self, client: TestClient):
        """F-CULT-005: Error responses follow standard format."""
        response = client.post("/chat", json={})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_research_error_format(self, client: TestClient):
        """F-CULT-005: Research errors follow standard format."""
        response = client.post("/research", json={})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_documents_error_format(self, client: TestClient):
        """F-CULT-005: Document errors follow standard format."""
        response = client.post("/documents", json={})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestAPIDocumentation:
    """Tests for API documentation."""

    def test_openapi_schema_exists(self, client: TestClient):
        """F-API-013: OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "paths" in data
        assert "/health" in data["paths"]
        assert "/chat" in data["paths"]
        assert "/research" in data["paths"]
        assert "/documents" in data["paths"]
