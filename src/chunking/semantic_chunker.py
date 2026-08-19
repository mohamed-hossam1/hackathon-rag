import logging
import re
from typing import List, Optional, Tuple

from src.chunking.base import Chunker
from src.config import get_config
from src.embedding.embedding_service import EmbeddingService
from src.models.chunk import Chunk, ChunkerType
from src.models.document import DocumentPage

logger = logging.getLogger("medical_rag.semantic_chunker")


class SemanticChunker(Chunker):
    """Semantic chunking strategy based on sentence embedding cosine similarity breakpoints."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        threshold: Optional[float] = None
    ):
        config = get_config()
        self.embedding_service = embedding_service or EmbeddingService()
        self.threshold = threshold if threshold is not None else config.SEMANTIC_CHUNK_THRESHOLD

    def _extract_sentences_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Extracts sentences from text along with their (start_char, end_char) offsets in text."""
        pattern = re.compile(r'\S.*?(?:[.!?]+(?=\s+|$)|(?:\r?\n)+|$)', re.DOTALL)
        sentences = []
        for match in pattern.finditer(text):
            raw = match.group(0)
            s_text = raw.strip()
            if not s_text:
                continue
            leading_ws = len(raw) - len(raw.lstrip())
            start = match.start() + leading_ws
            end = start + len(s_text)
            sentences.append((s_text, start, end))
        return sentences

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Computes cosine similarity between two vector embeddings."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def chunk(
        self,
        pages: List[DocumentPage],
        document_id: str,
        filename: str
    ) -> List[Chunk]:
        """Splits extracted document pages into a list of Chunk entities using semantic breakpoints.

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

        sentences = self._extract_sentences_with_spans(full_doc_text)
        if not sentences:
            logger.info(f"No sentences extracted for document {document_id}")
            return []

        if len(sentences) == 1:
            sentence_groups = [sentences]
        else:
            sentence_texts = [s[0] for s in sentences]
            embeddings = self.embedding_service.embed_texts(sentence_texts)

            sentence_groups = []
            current_group = [sentences[0]]

            for i in range(len(sentences) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                if sim < self.threshold:
                    sentence_groups.append(current_group)
                    current_group = [sentences[i + 1]]
                else:
                    current_group.append(sentences[i + 1])
            if current_group:
                sentence_groups.append(current_group)

        chunks: List[Chunk] = []
        for idx, group in enumerate(sentence_groups):
            start_char = group[0][1]
            end_char = group[-1][2]
            chunk_text = full_doc_text[start_char:end_char]

            page_start, page_end, method = self.get_chunk_metadata(start_char, end_char, page_offsets)

            chunk_id = f"{document_id}_semantic_{idx}"
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                document_id=document_id,
                filename=filename,
                page_start=page_start,
                page_end=page_end,
                chunk_index=idx,
                chunker_type=ChunkerType.SEMANTIC,
                method=method,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(chunk)

        logger.info(f"Generated {len(chunks)} semantic chunks for document {document_id}")
        return chunks
