"""Vector database for document storage and retrieval."""

import logging
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Singleton encoder to avoid reloading on every request
_encoder: SentenceTransformer | None = None


def get_encoder() -> SentenceTransformer:
    """Get or create the singleton sentence transformer encoder.

    Returns:
        SentenceTransformer instance (all-MiniLM-L6-v2, 384 dimensions).
    """
    global _encoder
    if _encoder is None:
        logger.info("Loading SentenceTransformer model (first load)")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


class VectorStore:
    """Vector database for document storage and retrieval.

    Uses Qdrant for vector storage and Sentence Transformers for embeddings.
    The embedding model (all-MiniLM-L6-v2) produces 384-dimensional vectors.

    Attributes:
        encoder: SentenceTransformer model for embeddings.
        client: Qdrant client instance.
        collection_name: Name of the vector collection.
    """

    # Expected embedding dimension for all-MiniLM-L6-v2
    EMBEDDING_DIM = 384

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_path: str | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            collection_name: Name for the vector collection. Each name is isolated.
            persist_path: Path for persistent storage. None uses in-memory (demo only).

        Note:
            In-memory storage is for prototypes only. Use persist_path in production.
        """
        self.encoder = get_encoder()
        self.collection_name = collection_name

        if persist_path:
            self.client = QdrantClient(path=persist_path)
            logger.info("VectorStore using persistent storage: %s", persist_path)
        else:
            self.client = QdrantClient(":memory:")
            logger.warning("VectorStore using in-memory storage (not for production)")

        self._create_collection()
        self._doc_counter = 0

    def _create_collection(self) -> None:
        """Initialize the vector collection."""
        dim = self.encoder.get_sentence_embedding_dimension()
        if dim != self.EMBEDDING_DIM:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %d",
                self.EMBEDDING_DIM,
                dim,
            )

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '%s' with %d dimensions",
            self.collection_name,
            dim,
        )

    def add_documents(self, documents: list[dict[str, Any]]) -> int:
        """Add documents to the vector store.

        Args:
            documents: List of document dicts with 'content' key (required)
                and optional 'title' and other metadata.

        Returns:
            Number of documents added.

        Raises:
            ValueError: If a document has empty content.
        """
        if not documents:
            return 0

        points = []
        for doc in documents:
            content = doc.get("content", "")
            if not content or not content.strip():
                logger.warning("Skipping document with empty content")
                continue

            # Truncate very long content (100k+ chars) to prevent memory issues
            if len(content) > 100000:
                logger.warning(
                    "Truncating document content from %d to 100000 chars",
                    len(content),
                )
                content = content[:100000]
                doc = {**doc, "content": content}

            vector = self.encoder.encode(content).tolist()
            points.append(
                models.PointStruct(
                    id=self._doc_counter,
                    vector=vector,
                    payload=doc,
                )
            )
            self._doc_counter += 1

        if points:
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info("Added %d documents to collection", len(points))

        return len(points)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for relevant documents.

        Args:
            query: Search query string.
            limit: Maximum number of results to return. Must be >= 1.

        Returns:
            List of document payloads ordered by relevance.
            Returns empty list if collection is empty or limit < 1.
        """
        if limit < 1:
            logger.warning("Invalid limit %d, returning empty results", limit)
            return []

        query_vector = self.encoder.encode(query).tolist()

        try:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
            )
            results = [hit.payload for hit in hits.points if hit.payload]
            logger.debug("Search returned %d results for query: %s", len(results), query[:50])
            return results
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    def count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            Number of documents stored.
        """
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
