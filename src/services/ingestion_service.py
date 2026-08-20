import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union


from src.chunking.recursive_chunker import RecursiveChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.cleaning.document_cleaner import DocumentCleaner
from src.cleaning.ocr_processor import OCRProcessor
from src.embedding.embedding_service import EmbeddingService
from src.models.document import DocumentPage, DocumentStatus, FileType
from src.parsers.base import Parser
from src.parsers.doc_parser import DOCParser
from src.parsers.docx_parser import DOCXParser
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.pdf_parser import PDFParser
from src.parsers.txt_parser import TXTParser
from src.retrieval.bm25_retriever import BM25Retriever
from src.services.document_store import DocumentStore
from src.vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger("medical_rag.services.ingestion")


class DocumentIngestionService:
    """Service for orchestrating document parsing, cleaning, OCR, chunking, embedding, vector storage, and BM25 index updating."""

    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_store: Optional[QdrantVectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        semantic_chunker: Optional[SemanticChunker] = None,
        recursive_chunker: Optional[RecursiveChunker] = None,
        document_cleaner: Optional[DocumentCleaner] = None,
        ocr_processor: Optional[OCRProcessor] = None,
        parsers: Optional[Mapping[str, Parser]] = None,
    ):
        self.document_store = document_store or DocumentStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_store = qdrant_store or QdrantVectorStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.semantic_chunker = semantic_chunker or SemanticChunker(embedding_service=self.embedding_service)
        self.recursive_chunker = recursive_chunker or RecursiveChunker()
        self.document_cleaner = document_cleaner or DocumentCleaner()
        self.ocr_processor = ocr_processor or OCRProcessor()

        self.parsers: Dict[str, Parser] = dict(parsers) if parsers is not None else {
            FileType.PDF.value: PDFParser(),
            FileType.DOCX.value: DOCXParser(),
            FileType.DOC.value: DOCParser(),
            FileType.MD.value: MarkdownParser(),
            FileType.TXT.value: TXTParser(),
        }

    def _get_parser_for_file(self, file_path: str, file_type: Optional[FileType] = None) -> Parser:
        """Selects appropriate parser instance based on FileType or file extension."""
        if file_type:
            ft_str = file_type.value if hasattr(file_type, "value") else str(file_type)
            if ft_str in self.parsers:
                return self.parsers[ft_str]

        ext = Path(file_path).suffix.lstrip(".").lower()
        if ext in self.parsers:
            return self.parsers[ext]

        raise ValueError(f"Unsupported file format: '{ext}' for file '{file_path}'")

    def ingest(self, document_id: str, file_path: str) -> bool:
        """Executes full ingestion pipeline for a document.

        Pipeline steps:
        1. Set document status to PROCESSING
        2. Select parser by file format and parse into raw DocumentPage list
        3. Execute OCR processor fallback on pages requiring OCR
        4. Execute DocumentCleaner pipeline
        5. Chunk text using both SemanticChunker and RecursiveChunker
        6. Embed all generated chunks with EmbeddingService
        7. Upsert chunks and vectors into QdrantVectorStore
        8. Save chunks into DocumentStore and rebuild BM25Retriever index across corpus
        9. Set document status to COMPLETED (or FAILED if error occurs)

        Args:
            document_id: UUID of document in DocumentStore.
            file_path: Local file path or storage path of target document.

        Returns:
            True if ingestion succeeded, False if failed.
        """
        logger.info(f"Starting ingestion pipeline for document_id='{document_id}' (file: '{file_path}')")

        document = self.document_store.get_document(document_id)
        if not document:
            err_msg = f"Document document_id='{document_id}' not found in DocumentStore"
            logger.error(err_msg)
            return False

        # Step 1: Update status to processing
        self.document_store.update_status(document_id, DocumentStatus.PROCESSING)

        try:
            # Step 2: Select parser & parse
            file_type = document.file_type
            parser = self._get_parser_for_file(file_path, file_type=file_type)
            raw_pages: List[DocumentPage] = parser.parse(file_path, document_id)
            logger.info(f"Parsed {len(raw_pages)} raw pages for document_id='{document_id}'")

            # Step 3: OCR fallback if PDF
            is_pdf = (file_type == FileType.PDF) or (Path(file_path).suffix.lower() == ".pdf")
            pdf_path_arg = file_path if is_pdf else None
            pages_after_ocr = self.ocr_processor.process_pages(raw_pages, pdf_path=pdf_path_arg)

            # Step 4: Clean pages
            cleaned_pages = self.document_cleaner.clean_pages(pages_after_ocr)

            # Step 5: Dual chunking strategy
            semantic_chunks = self.semantic_chunker.chunk(cleaned_pages, document_id, document.filename)
            recursive_chunks = self.recursive_chunker.chunk(cleaned_pages, document_id, document.filename)
            all_chunks = semantic_chunks + recursive_chunks

            logger.info(
                f"Chunked document_id='{document_id}': {len(semantic_chunks)} semantic + "
                f"{len(recursive_chunks)} recursive = {len(all_chunks)} total chunks"
            )

            # Step 6: Embed chunks
            all_vectors: List[List[float]] = []
            if all_chunks:
                chunk_texts = [c.text for c in all_chunks]
                all_vectors = self.embedding_service.embed_texts(chunk_texts)

            # Step 7: Upsert vectors to Qdrant
            if all_chunks and all_vectors:
                self.qdrant_store.upsert(all_chunks, all_vectors)

            # Step 8: Update DocumentStore chunks & rebuild BM25 index
            self.document_store.add_chunks(document_id, all_chunks)
            corpus_chunks = self.document_store.get_all_chunks()
            self.bm25_retriever.rebuild_index(corpus_chunks)

            # Step 9: Mark COMPLETED
            self.document_store.update_status(
                document_id,
                DocumentStatus.COMPLETED,
                total_pages=len(cleaned_pages)
            )
            logger.info(f"Successfully completed ingestion for document_id='{document_id}'")
            return True

        except Exception as err:
            logger.error(f"Failed ingestion for document_id='{document_id}': {err}", exc_info=True)
            self.document_store.update_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(err)
            )
            return False
        finally:
            is_primary_local_copy = False
            if document and document.storage_path and os.path.abspath(document.storage_path) == os.path.abspath(file_path):
                is_primary_local_copy = True

            if os.path.exists(file_path) and not is_primary_local_copy:
                try:
                    os.remove(file_path)
                    logger.info(f"Auto-cleaned temporary local file '{file_path}' after ingestion")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to remove temporary local file '{file_path}': {cleanup_err}")

