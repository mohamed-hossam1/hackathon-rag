import logging
import re
from typing import List, Optional, Set
from rank_bm25 import BM25Okapi

from src.models.chunk import Chunk
from src.models.retrieval import RetrievalMethod, RetrievalResult
from src.retrieval.base import Retriever

logger = logging.getLogger("medical_rag.retrieval.bm25")


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric lowercase tokenizer for BM25 indexing."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever(Retriever):
    """Keyword-based BM25 sparse retriever across all chunks."""

    def __init__(self, chunks: Optional[List[Chunk]] = None):
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        if chunks:
            self.build_index(chunks)

    def build_index(self, chunks: List[Chunk]) -> None:
        """Builds or rebuilds the BM25Okapi index from a list of chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25 = None
            self.tokenized_corpus = []
            logger.info("BM25 index cleared (empty chunks list provided)")
            return

        self.tokenized_corpus = [_tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Built BM25 index across {len(self.chunks)} chunks")

    def rebuild_index(self, chunks: List[Chunk]) -> None:
        """Rebuilds the BM25 index when new documents are added."""
        self.build_index(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieves top-K chunks matching query keywords using BM25.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult objects sorted by score descending.
        """
        if not query.strip() or not self.bm25 or not self.chunks:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        tokenized_query_set: Set[str] = set(tokenized_query)
        scores = self.bm25.get_scores(tokenized_query)

        # Sort indices by raw BM25 score descending
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results: List[RetrievalResult] = []
        for idx in scored_indices:
            if len(results) >= top_k:
                break

            chunk_tokens = set(self.tokenized_corpus[idx])
            matching_terms = chunk_tokens.intersection(tokenized_query_set)
            
            # Require at least one matching keyword
            if matching_terms:
                raw_score = float(scores[idx])
                # Ensure positive score for true keyword matches
                score = max(0.001, raw_score) + 0.1 * len(matching_terms)
                results.append(
                    RetrievalResult(
                        chunk=self.chunks[idx],
                        score=round(score, 4),
                        retrieval_method=RetrievalMethod.BM25
                    )
                )

        logger.info(f"BM25Retriever retrieved {len(results)} chunks for query='{query[:30]}...'")
        return results
