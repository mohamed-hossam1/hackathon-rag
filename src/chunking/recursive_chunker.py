import logging
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.base import Chunker
from src.config import get_config
from src.models.chunk import Chunk, ChunkerType
from src.models.document import DocumentPage

logger = logging.getLogger("medical_rag.recursive_chunker")


class RecursiveChunker(Chunker):
    """Recursive character chunking strategy using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        config = get_config()
        self.chunk_size = chunk_size if chunk_size is not None else config.RECURSIVE_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.RECURSIVE_CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "؟ ", "! ", "، ", " ", ""]
        )

    def chunk(
        self,
        pages: List[DocumentPage],
        document_id: str,
        filename: str
    ) -> List[Chunk]:
        """Splits extracted document pages into a list of Chunk entities using recursive character splitting.

        Args:
            pages: List of DocumentPage objects containing cleaned text and page metadata.
            document_id: Parent Document UUID string.
            filename: Original document filename.

        Returns:
            List of generated Chunk instances.
        """
        full_doc_text, page_offsets = self.prepare_document_text(pages)
        if not full_doc_text.strip():
            logger.info(f"No text to chunk for document {document_id}")
            return []

        raw_chunks = self.splitter.split_text(full_doc_text)
        if not raw_chunks:
            logger.info(f"No chunks generated for document {document_id}")
            return []

        chunks: List[Chunk] = []
        search_start = 0
        for idx, text in enumerate(raw_chunks):
            if not text:
                continue

            start = full_doc_text.find(text, search_start)
            if start == -1:
                start = full_doc_text.find(text)
                if start == -1:
                    start = search_start

            end = start + len(text)
            search_start = start

            page_start, page_end, method = self.get_chunk_metadata(start, end, page_offsets)

            chunk_id = f"{document_id}_recursive_{idx}"
            chunk = Chunk(
                chunk_id=chunk_id,
                text=text,
                document_id=document_id,
                filename=filename,
                page_start=page_start,
                page_end=page_end,
                chunk_index=idx,
                chunker_type=ChunkerType.RECURSIVE,
                method=method,
                start_char=start,
                end_char=end,
            )
            chunks.append(chunk)

        logger.info(f"Generated {len(chunks)} recursive chunks for document {document_id}")
        return chunks
