import logging
from typing import List, Optional

from src.config import get_config
from src.models.retrieval import RetrievalResult

logger = logging.getLogger("medical_rag.deduplication")


class ChunkDeduplicator:
    """Deduplicates candidate retrieval chunks using character-offset overlap ratios."""

    def __init__(self, threshold: Optional[float] = None):
        config = get_config()
        self.threshold = threshold if threshold is not None else config.DEDUP_OVERLAP_THRESHOLD

    def deduplicate(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicates a union list of candidate retrieval results based on character offset overlap.

        Args:
            candidates: List of RetrievalResult objects from all retrieval strategies.

        Returns:
            Deduplicated list of RetrievalResult objects.
        """
        if not candidates:
            return []

        # Sort candidates by score descending first
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        deduplicated: List[RetrievalResult] = []

        for candidate in sorted_candidates:
            chunk_a = candidate.chunk
            len_a = max(1, chunk_a.end_char - chunk_a.start_char)

            is_duplicate = False
            for existing in deduplicated:
                chunk_b = existing.chunk

                # Only check overlap between chunks from the SAME document
                if chunk_a.document_id == chunk_b.document_id:
                    len_b = max(1, chunk_b.end_char - chunk_b.start_char)
                    overlap_length = max(0, min(chunk_a.end_char, chunk_b.end_char) - max(chunk_a.start_char, chunk_b.start_char))
                    min_len = min(len_a, len_b)
                    overlap_ratio = overlap_length / min_len

                    if overlap_ratio >= self.threshold:
                        is_duplicate = True
                        logger.debug(
                            f"Duplicate detected (overlap={overlap_ratio:.2f} >= {self.threshold}): "
                            f"Dropping '{chunk_a.chunk_id}' in favor of '{chunk_b.chunk_id}'"
                        )
                        break

            if not is_duplicate:
                deduplicated.append(candidate)

        logger.info(f"Deduplicated candidate count from {len(candidates)} down to {len(deduplicated)} chunks")
        return deduplicated
