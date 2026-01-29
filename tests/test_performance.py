"""Performance tests - Section VII Falsification."""

import time


class TestPerformanceRAG:
    """Performance tests for RAG pipeline."""

    def test_rag_latency_under_200ms(self):
        """F-PERF-002: RAG retrieval must be < 200ms for 100 docs."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="perf_test")

        # Add 100 documents
        docs = [{"content": f"Document number {i} about topic {i % 10}"} for i in range(100)]
        store.add_documents(docs)

        # Measure search time
        start = time.time()
        results = store.search("document about topic 5", limit=5)
        elapsed = time.time() - start

        assert elapsed < 0.2, f"Search took {elapsed:.3f}s, must be < 0.2s"
        assert len(results) > 0

    def test_embedding_model_singleton(self):
        """F-PERF-004: Embedding model should load only once."""
        from src.rag.vectorstore import get_encoder

        # First call loads the model
        encoder1 = get_encoder()

        # Second call should return same instance
        encoder2 = get_encoder()

        assert encoder1 is encoder2, "Encoder should be singleton"

    def test_vectorstore_batch_performance(self):
        """F-RAG-014: Batch add should be efficient."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="batch_perf")

        docs = [{"content": f"Batch document {i}"} for i in range(50)]

        start = time.time()
        count = store.add_documents(docs)
        elapsed = time.time() - start

        assert count == 50
        assert elapsed < 5.0, f"Batch add took {elapsed:.2f}s, should be < 5s"


class TestPerformanceHTTP:
    """Performance tests for HTTP client pooling."""

    def test_http_client_singleton(self):
        """F-PERF-005: HTTP client should use connection pooling."""
        from src.agents.tools.web_search import get_http_client

        client1 = get_http_client()
        client2 = get_http_client()

        assert client1 is client2, "HTTP client should be singleton for pooling"
