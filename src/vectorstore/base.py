from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.models.chunk import Chunk
from src.models.retrieval import RetrievalResult


class VectorStore(ABC):
    """Abstract base class for vector database operations."""

    @abstractmethod
    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """Upserts a list of chunks and their corresponding embedding vectors into the vector store.

        Args:
            chunks: List of Chunk metadata models.
            vectors: List of embedding vector floats matching chunks length.

        Returns:
            True if upsert succeeded.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        chunker_type_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Performs vector similarity search against the store.

        Args:
            query_vector: Dense embedding vector for the search query.
            top_k: Maximum number of nearest neighbor results to return.
            chunker_type_filter: Optional filter by chunking strategy ('semantic' or 'recursive').

        Returns:
            List of RetrievalResult models sorted by similarity score descending.
        """
        pass

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """Deletes all vector points associated with a given document ID.

        Args:
            document_id: UUID of the document whose chunks should be deleted.

        Returns:
            True if deletion succeeded.
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Checks connectivity and operational status of the vector store cluster.

        Returns:
            Dictionary containing health status details.
        """
        pass
