import logging
from pathlib import Path
from typing import List, Optional, Union
import pymupdf as fitz  # PyMuPDF


from src.models.document import DocumentPage, ExtractionMethod
from src.parsers.base import Parser
from src.parsers.table_extractor import TableExtractor

logger = logging.getLogger("medical_rag.parsers.pdf_parser")

MIN_MEANINGFUL_CHAR_COUNT = 30


class PDFParser(Parser):
    """PDF Document Parser using PyMuPDF (fitz) with table extraction and OCR detection."""

    def __init__(self, table_extractor: Optional[TableExtractor] = None):
        self.table_extractor = table_extractor or TableExtractor()

    def parse(self, file_path: Union[str, Path], document_id: str) -> List[DocumentPage]:
        """Parses a PDF document file page-by-page into a list of DocumentPage models.

        Args:
            file_path: Path to the target PDF file.
            document_id: UUID of parent Document model.

        Returns:
            List of extracted DocumentPage instances.
        """
        path_str = str(file_path)
        if not Path(path_str).exists():
            raise FileNotFoundError(f"PDF file not found at path: '{path_str}'")

        pages: List[DocumentPage] = []

        try:
            logger.info(f"Opening PDF file '{path_str}' for document_id='{document_id}'")
            doc = fitz.open(path_str)

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1

                # 1. Extract raw text page
                extracted = page.get_text("text")
                raw_text = str(extracted) if isinstance(extracted, str) else ""
                stripped_text = raw_text.strip()

                # 2. Check for meaningful text (detect scanned image pages for OCR fallback)
                has_meaningful_text = len(stripped_text) >= MIN_MEANINGFUL_CHAR_COUNT
                extraction_method = (
                    ExtractionMethod.NATIVE if has_meaningful_text else ExtractionMethod.OCR
                )

                # 3. Extract tables using TableExtractor
                tables = self.table_extractor.extract_from_pymupdf_page(page)

                pages.append(
                    DocumentPage(
                        document_id=document_id,
                        page_number=page_number,
                        raw_text=raw_text,
                        cleaned_text=raw_text,  # Cleaned by DocumentCleaner in downstream pipeline
                        extraction_method=extraction_method,
                        tables=tables,
                        has_meaningful_text=has_meaningful_text
                    )
                )

            doc.close()
            logger.info(f"Parsed PDF '{path_str}': {len(pages)} pages extracted")
        except Exception as err:
            logger.error(f"Failed to parse PDF file '{path_str}': {err}")
            raise RuntimeError(f"Failed to parse PDF document: {str(err)}") from err

        return pages
