import logging
import re
from typing import List

from src.models.document import DocumentPage

logger = logging.getLogger("medical_rag.cleaning.document_cleaner")

# Remove non-printable control characters except line feeds and tabs
NON_PRINTABLE_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class DocumentCleaner:
    """Cleaner for normalizing extracted document page text while preserving Markdown structure."""

    def clean_text(self, text: str) -> str:
        """Cleans raw text string by removing non-printable characters and normalizing whitespace.

        Args:
            text: Raw text string.

        Returns:
            Cleaned and normalized text string.
        """
        if not text:
            return ""

        # 1. Strip non-printable control characters
        text = NON_PRINTABLE_PATTERN.sub("", text)

        # 2. Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Collapse 3+ consecutive newlines to 2 newlines (preserve paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4. Strip trailing line spaces without modifying table or list indents
        lines = [line.rstrip() for line in text.split("\n")]
        cleaned = "\n".join(lines).strip()

        return cleaned

    def clean_page(self, page: DocumentPage) -> DocumentPage:
        """Cleans a single DocumentPage text and returns an updated copy.

        Args:
            page: Input DocumentPage model.

        Returns:
            Updated DocumentPage with cleaned_text populated.
        """
        cleaned_text = self.clean_text(page.raw_text)

        # Append normalized table markdown if tables extracted and not already present
        if page.tables:
            table_markdowns = [t.normalized_text for t in page.tables if t.normalized_text]
            for table_md in table_markdowns:
                if table_md and table_md not in cleaned_text:
                    cleaned_text += f"\n\n{table_md}"

        # Re-verify meaningful text after cleaning
        has_meaningful_text = len(cleaned_text.strip()) >= 30

        return page.model_copy(
            update={
                "cleaned_text": cleaned_text,
                "has_meaningful_text": has_meaningful_text
            }
        )

    def clean_pages(self, pages: List[DocumentPage]) -> List[DocumentPage]:
        """Cleans a list of DocumentPage models sequentially.

        Args:
            pages: List of DocumentPage instances.

        Returns:
            List of cleaned DocumentPage instances.
        """
        cleaned_pages = [self.clean_page(p) for p in pages]
        logger.info(f"Cleaned {len(cleaned_pages)} document pages")
        return cleaned_pages
