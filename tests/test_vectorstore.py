"""Tests for the VectorStore RAG component."""


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""

    def test_collection_isolation(self):
        """F-RAG-001: Collections with different names are isolated."""
        from src.rag.vectorstore import VectorStore

        store1 = VectorStore(collection_name="test1")
        store2 = VectorStore(collection_name="test2")

        store1.add_documents([{"content": "Document in store 1"}])

        # Store 2 should be empty
        assert store2.count() == 0
        assert store1.count() == 1

    def test_embedding_dimension_matches(self):
        """F-RAG-012: Embedding dimension matches Qdrant config (384 for MiniLM)."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="dim_test")
        assert store.EMBEDDING_DIM == 384
        assert store.encoder.get_sentence_embedding_dimension() == 384


class TestVectorStoreSearch:
    """Tests for VectorStore search functionality."""

    def test_search_empty_store_returns_empty(self):
        """F-RAG-002: Search empty VectorStore returns empty list."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="empty_test")
        results = store.search("any query")
        assert results == []

    def test_search_limit_works(self):
        """F-RAG-010: Search limit parameter works correctly."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="limit_test")
        store.add_documents(
            [
                {"content": "Document 1"},
                {"content": "Document 2"},
                {"content": "Document 3"},
            ]
        )

        results = store.search("document", limit=1)
        assert len(results) == 1

    def test_search_invalid_limit(self):
        """F-RAG-011: Search with limit < 1 returns empty."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="invalid_limit")
        store.add_documents([{"content": "Test document"}])

        assert store.search("test", limit=0) == []
        assert store.search("test", limit=-1) == []

    def test_search_exact_match_ranks_first(self):
        """F-RAG-006: Exact match should rank highly."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="exact_match")
        store.add_documents(
            [
                {"title": "Machine Learning Basics", "content": "Introduction to ML"},
                {"title": "Deep Learning Guide", "content": "Neural networks explained"},
                {"title": "Python Tutorial", "content": "Learn Python programming"},
            ]
        )

        results = store.search("machine learning", limit=3)
        # First result should be most relevant
        assert len(results) > 0
        assert "Machine Learning" in results[0].get("title", "") or "ML" in results[0].get("content", "")


class TestVectorStoreDocuments:
    """Tests for VectorStore document operations."""

    def test_add_empty_content_skipped(self):
        """F-RAG-003: Documents with empty content are skipped."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="empty_content")
        count = store.add_documents([{"content": ""}])
        assert count == 0
        assert store.count() == 0

    def test_add_large_document_truncated(self):
        """F-RAG-004: Very large documents are truncated."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="large_doc")
        large_content = "x" * 150000  # 150k chars

        count = store.add_documents([{"content": large_content}])
        assert count == 1  # Should still be added

    def test_metadata_preserved(self):
        """F-RAG-015: Document metadata is preserved and returned."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="metadata_test")
        store.add_documents(
            [
                {
                    "title": "Test Doc",
                    "content": "Test content",
                    "author": "Test Author",
                    "tags": ["test", "example"],
                }
            ]
        )

        results = store.search("test")
        assert len(results) == 1
        assert results[0]["title"] == "Test Doc"
        assert results[0]["author"] == "Test Author"
        assert results[0]["tags"] == ["test", "example"]

    def test_special_characters_handled(self):
        """F-RAG-013: Special characters and emoji in content work."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="special_chars")
        store.add_documents(
            [
                {
                    "content": "Test with emoji 🚀 and special chars: <>&\"'",
                }
            ]
        )

        results = store.search("emoji")
        assert len(results) == 1
        assert "🚀" in results[0]["content"]

    def test_batch_add_documents(self):
        """F-RAG-014: Batch document addition works."""
        from src.rag.vectorstore import VectorStore

        store = VectorStore(collection_name="batch_test")
        docs = [{"content": f"Document {i}"} for i in range(10)]

        count = store.add_documents(docs)
        assert count == 10
        assert store.count() == 10
