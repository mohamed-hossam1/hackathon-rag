import logging
import re
from typing import List, Optional

from src.config import get_config
from src.models.retrieval import RetrievalResult

logger = logging.getLogger("medical_rag.deduplication")


def _compute_text_overlap(text1: str, text2: str) -> float:
    """Computes word-level overlap ratio relative to the smaller chunk."""
    w1 = set(w.lower() for w in re.findall(r"\w+", text1) if len(w) > 2)
    w2 = set(w.lower() for w in re.findall(r"\w+", text2) if len(w) > 2)
    if not w1 or not w2:
        return 0.0
    intersection = w1.intersection(w2)
    min_count = min(len(w1), len(w2))
    return len(intersection) / min_count


class ChunkDeduplicator:
    """Deduplicates candidate retrieval chunks using character-offset and text word-overlap ratios."""

    def __init__(self, threshold: Optional[float] = None):
        config = get_config()
        self.threshold = threshold if threshold is not None else config.DEDUP_OVERLAP_THRESHOLD

    def deduplicate(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicates a union list of candidate retrieval results based on offset and text overlap.

        Args:
            candidates: List of RetrievalResult objects from all retrieval strategies.

        Returns:
            Deduplicated list of RetrievalResult objects.
        """
        if not candidates:
            return []

        def _get_sort_key(res: RetrievalResult) -> float:
            fusion_val = getattr(res, "_fusion_score", None)
            if isinstance(fusion_val, (int, float)):
                return float(fusion_val)
            return float(res.score)

        # Sort candidates by fusion score (RRF) or raw score descending
        sorted_candidates = sorted(candidates, key=_get_sort_key, reverse=True)
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
                    char_overlap_ratio = overlap_length / min_len

                    text_overlap_ratio = _compute_text_overlap(chunk_a.text, chunk_b.text)

                    if char_overlap_ratio >= self.threshold or text_overlap_ratio >= 0.65:
                        is_duplicate = True
                        logger.debug(
                            f"Duplicate detected (char_overlap={char_overlap_ratio:.2f}, text_overlap={text_overlap_ratio:.2f}): "
                            f"Dropping '{chunk_a.chunk_id}' in favor of '{chunk_b.chunk_id}'"
                        )
                        break

            if not is_duplicate:
                deduplicated.append(candidate)

        logger.info(f"Deduplicated candidate count from {len(candidates)} down to {len(deduplicated)} chunks")
        return deduplicated
