import logging
from typing import List, Optional

from src.embedding.embedding_service import EmbeddingService
from src.models.retrieval import RetrievalResult
from src.retrieval.base import Retriever
from src.vectorstore.base import VectorStore
from src.vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger("medical_rag.retrieval.recursive")


class RecursiveRetriever(Retriever):
    """Dense vector retriever targeting recursive character chunking strategy."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or QdrantVectorStore()

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieves top-K recursive chunks matching the query vector.

        Args:
            query: User search query text.
            top_k: Number of candidates to retrieve.

        Returns:
            List of RetrievalResult objects with scores.
        """
        if not query.strip():
            return []

        query_vector = self.embedding_service.embed_query(query)
        if not query_vector:
            logger.warning("Query embedding returned empty vector")
            return []

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            chunker_type_filter="recursive"
        )
        logger.info(f"RecursiveRetriever retrieved {len(results)} chunks for query='{query[:30]}...'")
        return results
