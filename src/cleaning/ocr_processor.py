import logging
from typing import List, Optional

from src.models.document import DocumentPage, ExtractionMethod

logger = logging.getLogger("medical_rag.cleaning.ocr_processor")


class OCRProcessor:
    """OCR Fallback Processor using pytesseract and pdf2image for scanned PDF pages."""

    def process_page(
        self,
        page: DocumentPage,
        pdf_path: Optional[str] = None
    ) -> DocumentPage:
        """Processes a single DocumentPage with OCR if page lacks meaningful native text.

        Args:
            page: DocumentPage instance to check.
            pdf_path: Optional file path to parent PDF file for image rendering.

        Returns:
            Updated DocumentPage object.
        """
        if page.has_meaningful_text:
            return page

        if not pdf_path:
            logger.warning(
                f"Page #{page.page_number} lacks meaningful text, but no pdf_path provided for OCR rendering."
            )
            return page

        logger.info(
            f"Attempting OCR fallback for page #{page.page_number} (document_id='{page.document_id}')"
        )

        try:
            from pdf2image import convert_from_path
            import pytesseract

            # Render PDF page to PIL Image
            images = convert_from_path(
                pdf_path,
                first_page=page.page_number,
                last_page=page.page_number
            )

            if not images:
                logger.warning(f"pdf2image returned no image for page #{page.page_number}")
                return page

            img = images[0]
            extracted_ocr = pytesseract.image_to_string(img)
            ocr_text = str(extracted_ocr) if isinstance(extracted_ocr, str) else ""
            stripped_ocr = ocr_text.strip()

            has_text = len(stripped_ocr) >= 30
            logger.info(
                f"OCR executed on page #{page.page_number}: {len(stripped_ocr)} chars extracted"
            )

            return page.model_copy(
                update={
                    "raw_text": ocr_text if has_text else page.raw_text,
                    "cleaned_text": ocr_text if has_text else page.cleaned_text,
                    "extraction_method": ExtractionMethod.OCR,
                    "has_meaningful_text": has_text or page.has_meaningful_text
                }
            )

        except (ImportError, FileNotFoundError) as sys_err:
            logger.warning(
                f"OCR dependencies (tesseract/poppler) not available on system: {sys_err}. "
                f"Skipping OCR for page #{page.page_number}."
            )
            return page
        except Exception as err:
            logger.error(
                f"Failed to execute OCR on page #{page.page_number} of '{pdf_path}': {err}"
            )
            return page

    def process_pages(
        self,
        pages: List[DocumentPage],
        pdf_path: Optional[str] = None
    ) -> List[DocumentPage]:
        """Processes a list of DocumentPage models with OCR fallback.

        Args:
            pages: List of DocumentPage instances.
            pdf_path: Path to parent PDF file.

        Returns:
            List of updated DocumentPage instances.
        """
        processed = [self.process_page(p, pdf_path=pdf_path) for p in pages]
        return processed
