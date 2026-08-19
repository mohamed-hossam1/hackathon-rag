from abc import ABC, abstractmethod
from typing import List, Tuple

from src.models.chunk import Chunk
from src.models.document import DocumentPage, ExtractionMethod


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

    def prepare_document_text(
        self, pages: List[DocumentPage]
    ) -> Tuple[str, List[Tuple[int, int, int, str]]]:
        """Concatenates cleaned page text and tracks character offset ranges per page.

        Returns:
            Tuple of (full_doc_text, page_offsets) where page_offsets is a list of
            tuples: (page_number, start_offset, end_offset, extraction_method_str).
        """
        page_offsets = []
        full_text_parts = []
        current_offset = 0

        for page in pages:
            text = page.cleaned_text
            if not text:
                continue
            if full_text_parts:
                full_text_parts.append("\n\n")
                current_offset += 2

            start_offset = current_offset
            full_text_parts.append(text)
            current_offset += len(text)
            end_offset = current_offset

            method_str = (
                page.extraction_method.value
                if hasattr(page.extraction_method, "value")
                else str(page.extraction_method)
            )
            page_offsets.append((page.page_number, start_offset, end_offset, method_str))

        full_doc_text = "".join(full_text_parts)
        return full_doc_text, page_offsets

    def get_chunk_metadata(
        self,
        start_char: int,
        end_char: int,
        page_offsets: List[Tuple[int, int, int, str]]
    ) -> Tuple[int, int, str]:
        """Maps character offsets in full_doc_text to (page_start, page_end, method)."""
        if not page_offsets:
            return 1, 1, "native"

        overlapping = [
            p for p in page_offsets
            if max(start_char, p[1]) < min(end_char, p[2])
        ]

        if not overlapping:
            # Fallback to closest page by character distance
            closest = min(
                page_offsets,
                key=lambda p: min(abs(p[1] - start_char), abs(p[2] - end_char))
            )
            return closest[0], closest[0], closest[3]

        page_start = overlapping[0][0]
        page_end = overlapping[-1][0]
        has_ocr = any(p[3] == ExtractionMethod.OCR.value or p[3] == "ocr" for p in overlapping)
        method = "ocr" if has_ocr else "native"

        return page_start, page_end, method

