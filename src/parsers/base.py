from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from src.models.document import DocumentPage


class Parser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: Union[str, Path], document_id: str) -> List[DocumentPage]:
        """Parses a document file and extracts pages with raw text, cleaned text, and table data.

        Args:
            file_path: Path to the target document file.
            document_id: UUID string of the parent Document entity.

        Returns:
            List of DocumentPage objects extracted from the document.
        """
        pass
