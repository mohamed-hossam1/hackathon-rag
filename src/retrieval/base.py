from abc import ABC, abstractmethod
from typing import List

from src.models.retrieval import RetrievalResult


class Retriever(ABC):
    """Abstract base class for information retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieves top-K relevant chunks for a user query.

        Args:
            query: User search query text.
            top_k: Maximum number of relevant chunks to retrieve.

        Returns:
            List of RetrievalResult objects sorted by score descending.
        """
        pass
