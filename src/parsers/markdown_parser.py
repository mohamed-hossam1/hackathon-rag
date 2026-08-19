import logging
from pathlib import Path
from typing import List, Union

from src.models.document import DocumentPage, ExtractionMethod
from src.parsers.base import Parser

logger = logging.getLogger("medical_rag.parsers.markdown_parser")

MIN_MEANINGFUL_CHAR_COUNT = 30


class MarkdownParser(Parser):
    """Markdown (.md) Document Parser preserving heading structure."""

    def parse(self, file_path: Union[str, Path], document_id: str) -> List[DocumentPage]:
        """Parses a Markdown document file into a single DocumentPage instance.

        Args:
            file_path: Path to target Markdown file.
            document_id: UUID of parent Document entity.

        Returns:
            List containing 1 DocumentPage.
        """
        path_str = str(file_path)
        path_obj = Path(path_str)

        if not path_obj.exists():
            raise FileNotFoundError(f"Markdown file not found at path: '{path_str}'")

        try:
            logger.info(f"Reading Markdown file '{path_str}' for document_id='{document_id}'")
            try:
                raw_text = path_obj.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for '{path_str}', falling back to latin-1")
                raw_text = path_obj.read_text(encoding="latin-1")

            stripped_text = raw_text.strip()
            has_meaningful_text = len(stripped_text) >= MIN_MEANINGFUL_CHAR_COUNT
            extraction_method = (
                ExtractionMethod.NATIVE if has_meaningful_text else ExtractionMethod.OCR
            )

            page = DocumentPage(
                document_id=document_id,
                page_number=1,
                raw_text=raw_text,
                cleaned_text=raw_text,
                extraction_method=extraction_method,
                tables=[],
                has_meaningful_text=has_meaningful_text
            )

            logger.info(f"Parsed Markdown '{path_str}': single page extracted ({len(raw_text)} chars)")
            return [page]
        except Exception as err:
            logger.error(f"Failed to parse Markdown file '{path_str}': {err}")
            raise RuntimeError(f"Failed to parse Markdown document: {str(err)}") from err
