from abc import ABC, abstractmethod
from typing import List

from src.models.chunk import Chunk
from src.models.document import DocumentPage


class Chunker(ABC):
    """Abstract base class for document text chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        pages: List[DocumentPage],
        document_id: str,
        filename: str
    ) -> List[Chunk]:
        """Splits extracted document pages into a list of Chunk entities.

        Args:
            pages: List of DocumentPage objects containing cleaned text and page metadata.
            document_id: Parent Document UUID string.
            filename: Original document filename for lineage tracking.

        Returns:
            List of generated Chunk instances.
        """
        pass
