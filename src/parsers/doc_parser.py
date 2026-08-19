import logging
from pathlib import Path
import subprocess
from typing import List, Union

from src.models.document import DocumentPage, ExtractionMethod
from src.parsers.base import Parser

logger = logging.getLogger("medical_rag.parsers.doc_parser")

MIN_MEANINGFUL_CHAR_COUNT = 30


class DOCParser(Parser):
    """Legacy DOC (.doc) Document Parser using antiword CLI tool."""

    def parse(self, file_path: Union[str, Path], document_id: str) -> List[DocumentPage]:
        """Parses a legacy DOC file using antiword shell process.

        Args:
            file_path: Path to target legacy .doc file.
            document_id: UUID of parent Document entity.

        Returns:
            List containing 1 DocumentPage.
        """
        path_str = str(file_path)
        if not Path(path_str).exists():
            raise FileNotFoundError(f"DOC file not found at path: '{path_str}'")

        try:
            logger.info(f"Opening DOC file '{path_str}' via antiword for document_id='{document_id}'")
            res = subprocess.run(
                ["antiword", path_str],
                capture_output=True,
                text=True,
                check=False
            )

            if res.returncode != 0:
                err_msg = res.stderr.strip() or "antiword process returned non-zero exit code"
                logger.error(f"antiword failed on '{path_str}': {err_msg}")
                raise RuntimeError(f"antiword text extraction failed: {err_msg}")

            raw_text = res.stdout or ""
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

            logger.info(f"Parsed DOC '{path_str}': single page extracted ({len(raw_text)} chars)")
            return [page]
        except FileNotFoundError as fnf:
            logger.error("antiword executable not found on system path")
            raise RuntimeError(
                "Legacy .doc parser requires 'antiword' utility which is not installed on this system. "
                "Please install antiword (e.g., sudo apt-get install antiword)."
            ) from fnf
        except Exception as err:
            logger.error(f"Failed to parse DOC file '{path_str}': {err}")
            raise RuntimeError(f"Failed to parse DOC document: {str(err)}") from err
